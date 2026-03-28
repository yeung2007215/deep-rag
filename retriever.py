"""
retriever.py — 檢索與重排模組
"""
import logging
import re
from typing import Optional

from langchain.agents import create_agent
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL,
    SIMILARITY_SEARCH_K, BM25_SEARCH_K, FINAL_TOP_K,
    DEEPRAG_MAX_ROUNDS, DEEPRAG_FOLLOWUP_N, DEEPRAG_MIN_CONTEXT_LENGTH,
    RERANK_TOP_K, RERANK_CONFIDENCE_THRESHOLD,
    CHAT_HISTORY_MAX_TURNS, CHAT_HISTORY_ANSWER_MAX_CHARS,
    MAX_CONTEXT_CHARS, CANTONESE_TERM_MAP,
)

logger = logging.getLogger(__name__)

# ==============================
# 對話歷史型別定義
# ==============================
# ChatHistory = [(使用者問題, AI 回答), ...]，最舊的在前
ChatHistory = list[tuple[str, str]]


def _format_chat_history(chat_history: ChatHistory) -> str:
    """
    將對話歷史格式化為 LLM prompt 用的字串。
    保護機制：只取最近 N 輪，且每輪回答截斷至 CHAT_HISTORY_ANSWER_MAX_CHARS。
    """
    if not chat_history:
        return ""
    recent = chat_history[-CHAT_HISTORY_MAX_TURNS:]
    lines = []
    for i, (q, a) in enumerate(recent, 1):
        lines.append(f"[輪{i}] 使用者：{q}")
        truncated_a = a[:CHAT_HISTORY_ANSWER_MAX_CHARS]
        if len(a) > CHAT_HISTORY_ANSWER_MAX_CHARS:
            truncated_a += "…（截斷）"
        lines.append(f"[輪{i}] 助理：{truncated_a}")
    return "\n".join(lines)


def _has_cantonese(text: str) -> bool:
    """快速偵測文本是否包含粵語用詞"""
    cantonese_markers = [
        "點樣", "點先", "幾多", "攞", "擺", "掟", "揀", "嘅", "佢",
        "咗", "唔", "冇", "係", "嗰個", "嗰啲", "呢個", "乜嘢", "邊個", "幾時",
    ]
    return any(marker in text for marker in cantonese_markers)


def expand_cantonese_terms(query: str) -> list[str]:
    """
    利用術語對照表，將粵語關鍵字展開成書面語同義詞。
    回傳展開後的額外 queries（原 query 不修改）。
    用途：增強 BM25 關鍵字檢索在粵語環境下的命中率。
    """
    expanded_queries = []
    for cantonese_term, standard_terms in CANTONESE_TERM_MAP.items():
        if cantonese_term in query:
            for std in standard_terms:
                new_q = query.replace(cantonese_term, std)
                if new_q != query and new_q not in expanded_queries:
                    expanded_queries.append(new_q)
    return expanded_queries


def rewrite_query_with_history(question: str, chat_history: ChatHistory) -> str:
    """
    Query 改寫：指代消解 + 廣東話轉書面語。

    功能：
    1. 指代消解：「那三國殺呢？」→「三國殺的起始手牌是幾張？」
    2. 粵語轉換：「點樣先贏？」→「勝利條件是什麼？」
    """
    # 判斷是否需要改寫
    anaphora_indicators = ["那", "它", "這", "呢", "如果是", "同樣", "也", "又", "換"]
    is_short = len(question) <= 20
    has_anaphora = any(ind in question for ind in anaphora_indicators)
    has_cantonese = _has_cantonese(question)
    has_history = bool(chat_history)

    # 需要改寫的情況：有歷史+有指代 or 有歷史+太短 or 有粵語
    needs_rewrite = (has_history and (has_anaphora or is_short)) or has_cantonese
    if not needs_rewrite:
        return question

    history_str = _format_chat_history(chat_history) if chat_history else ""

    # 構建 prompt（根據情境調整指令）
    rules = [
        "1. 如果問題包含廣東話/粵語口語，改寫成標準中文書面語（例：「點樣先贏」→「勝利條件是什麼」）",
        "2. 如果問題有指代詞（那、它、這、呢等），用對話歷史中的具體主詞替換",
        "3. 保留使用者意圖中的新增條件或轉折",
        "4. 如果問題已完整清晰（非口語、無指代），直接回傳原問題",
        "5. 只輸出改寫後的問題，不要解釋、不要加引號",
    ]
    prompt_sys = "你是查詢改寫助理。將使用者的口語化或模糊問題改寫成適合檢索規則資料庫的標準中文。\n\n規則：\n" + "\n".join(rules)

    user_content = f"使用者問題：{question}"
    if history_str:
        user_content = f"對話歷史：\n{history_str}\n\n{user_content}"

    try:
        agent = create_agent(model=LLM_MODEL, system_prompt=prompt_sys)
        r = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
        rewritten = r["messages"][-1].content.strip()
        if rewritten and rewritten != question:
            logger.info(f"Query rewrite: 「{question}」→ 「{rewritten}」")
        return rewritten or question
    except Exception as e:
        logger.warning(f"Query rewrite 失敗，使用原問題: {e}")
        return question


def get_vector_store(collection_name: str) -> Chroma:
    """取得指定 collection 的 ChromaDB 向量資料庫實例"""
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_all_documents(vector_store: Chroma) -> list[Document]:
    """從 ChromaDB 取出所有文件，用於建立 BM25 索引"""
    try:
        collection = vector_store._collection
        result = collection.get(include=["documents", "metadatas"])
        docs = []
        if result and result.get("documents"):
            metas = result.get("metadatas") or [{}] * len(result["documents"])
            for text, meta in zip(result["documents"], metas):
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata=meta or {}))
        return docs
    except Exception as e:
        logger.warning(f"無法取出文件建立 BM25 索引: {e}")
        return []


def build_bm25_retriever(
    bm25_docs: list[Document], k: int = BM25_SEARCH_K
) -> Optional[BM25Retriever]:
    """預先建立 BM25 索引（只需建一次，供多次查詢複用）"""
    if not bm25_docs:
        return None
    try:
        retriever = BM25Retriever.from_documents(bm25_docs)
        retriever.k = k
        return retriever
    except Exception as e:
        logger.warning(f"BM25 索引建立失敗: {e}")
        return None


def hybrid_search(
    query: str, vector_store: Chroma,
    bm25_retriever: Optional[BM25Retriever] = None,
    k_vector: int = SIMILARITY_SEARCH_K,
    final_k: int = FINAL_TOP_K,
) -> list[Document]:
    """結合 BM25 + 向量語意檢索，去重合併"""
    try:
        vector_results = vector_store.similarity_search(query, k=k_vector)
    except Exception as e:
        logger.error(f"向量搜尋失敗: {e}")
        vector_results = []

    bm25_results = []
    if bm25_retriever:
        try:
            bm25_results = bm25_retriever.invoke(query)
        except Exception as e:
            logger.warning(f"BM25 搜尋失敗: {e}")

    seen, merged = set(), []
    for doc in vector_results + bm25_results:
        key = doc.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(doc)
        if len(merged) >= final_k:
            break
    logger.info(f"混合檢索: vector={len(vector_results)}, bm25={len(bm25_results)}, merged={len(merged)}")
    return merged


def _parse_rerank_output(text: str, n_docs: int, threshold: float) -> list[tuple[int, float]]:
    picked, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[,，\s]+", line)
        if not parts or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        try:
            conf = float(parts[1]) / 100.0 if len(parts) >= 2 else 1.0
        except ValueError:
            conf = 1.0
        if 1 <= idx <= n_docs and idx not in seen and conf >= threshold:
            picked.append((idx, conf))
            seen.add(idx)
    return picked


def llm_rerank(
    question: str, docs: list[Document],
    top_k: int = RERANK_TOP_K,
    confidence_threshold: float = RERANK_CONFIDENCE_THRESHOLD,
) -> list[Document]:
    """LLM Reranker：metadata 感知 + 信心門檻"""
    if not docs:
        return []

    blocks = []
    for i, d in enumerate(docs, start=1):
        text = d.page_content.replace("\n", " ").strip()[:450]
        game = d.metadata.get("game_name", "未知")
        blocks.append(f"[{i}] (遊戲:{game}) {text}")

    thr = int(confidence_threshold * 100)
    rerank_prompt = (
        f"你是桌遊規則檢索重排器。從候選段落中選出最能回答問題的段落（最多 {top_k} 段）。\n\n"
        f"評分標準（0-100）：\n"
        f"  90-100：段落直接回答了問題的核心\n"
        f"  60-89：段落包含相關規則但需要推理\n"
        f"  30-59：段落僅間接相關\n"
        f"  0-29：段落與問題無關（不要輸出）\n\n"
        f"注意：\n"
        f"- 候選段落標記了所屬遊戲名稱，請優先選擇與問題相關遊戲的段落\n"
        f"- 如果問題指定了某個遊戲，其他遊戲的段落信心應大幅降低\n"
        f"- 只保留信心 ≥{thr} 的段落\n\n"
        f"輸出格式（每行一段，不要其他文字）：編號,信心"
    )
    try:
        agent = create_agent(model=LLM_MODEL, system_prompt=rerank_prompt)
        prompt = f"問題：{question}\n\n候選段落：\n" + "\n".join(blocks)
        r = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        out = r["messages"][-1].content.strip()
    except Exception as e:
        logger.error(f"LLM Rerank 失敗: {e}")
        return docs[:top_k]

    picked = _parse_rerank_output(out, len(docs), confidence_threshold)
    picked.sort(key=lambda x: x[1], reverse=True)
    result = [docs[idx - 1] for idx, _ in picked[:top_k]]
    logger.info(f"Rerank: {len(result)}/{len(docs)} 段通過門檻 (>={thr}%)")
    return result


# ==============================
# Context 充足性判斷（LLM 驅動）
# ==============================
def is_context_sufficient(question: str, context: str) -> bool:
    """使用 LLM 判斷檢索內容是否足以回答問題"""
    if not context or len(context.strip()) < DEEPRAG_MIN_CONTEXT_LENGTH:
        logger.info(f"Context 長度不足 ({len(context.strip()) if context else 0})")
        return False

    prompt_sys = (
        "你是檢索品質評估器。判斷檢索內容是否足以回答問題。\n"
        "找出問題中的關鍵實體，檢查是否涵蓋。\n"
        "只輸出一行：\nSUFFICIENT\n或\nINSUFFICIENT:缺少XXX"
    )
    try:
        agent = create_agent(model=LLM_MODEL, system_prompt=prompt_sys)
        r = agent.invoke({"messages": [{"role": "user",
            "content": f"問題：{question}\n\n檢索內容：\n{context[:2000]}"}]})
        answer = r["messages"][-1].content.strip()
        is_ok = answer.upper().startswith("SUFFICIENT") and "INSUFFICIENT" not in answer.upper()
        logger.info(f"充足性: {'✅' if is_ok else '❌'} — {answer[:80]}")
        return is_ok
    except Exception as e:
        logger.warning(f"充足性判斷失敗，降級為長度判斷: {e}")
        return len(context.strip()) >= DEEPRAG_MIN_CONTEXT_LENGTH * 3


def generate_followup_queries(
    question: str,
    context: str,
    n: int = DEEPRAG_FOLLOWUP_N,
    chat_history: Optional[ChatHistory] = None,
) -> list[str]:
    """
    用 LLM 產生更精準的 follow-up 檢索 query。
    若提供 chat_history，可利用前幾輪對話的主題補強 query。
    """
    prompt_sys = (
        f"你是檢索 query 生成器。為桌遊規則資料庫生成更精準的查詢。\n"
        f"輸出 {n} 條中文 query，每條一行，不加編號不解釋。\n"
        f"貼近規則書用語、嘗試同義詞、包含遊戲名稱。"
    )
    history_section = ""
    if chat_history:
        history_section = f"\n\n對話歷史（供參考，勿重複已問過的問題）：\n{_format_chat_history(chat_history)}"

    try:
        agent = create_agent(model=LLM_MODEL, system_prompt=prompt_sys)
        r = agent.invoke({"messages": [{"role": "user",
            "content": (
                f"原始問題：{question}\n\n"
                f"目前檢索內容：\n{context[:2000]}"
                f"{history_section}"
            )}]})
        text = r["messages"][-1].content.strip()
        queries = [l.strip() for l in text.splitlines() if l.strip()]
        return queries[:n] if queries else []
    except Exception as e:
        logger.error(f"Follow-up query 生成失敗: {e}")
        return []


# ==============================
# DeepRAG 主流程
# ==============================
def deep_rag_retrieve(
    question: str,
    vector_store: Chroma,
    bm25_retriever: Optional[BM25Retriever] = None,
    chat_history: Optional[ChatHistory] = None,
) -> tuple[str, list[str], list[Document]]:
    """
    DeepRAG: 多輪迭代檢索 + 混合搜尋 + Rerank

    Args:
        bm25_retriever: 預建的 BM25 索引（用 build_bm25_retriever 建立）
        chat_history: 對話歷史，用於指代消解

    Returns: (final_context, all_queries, reranked_docs)
    其中 all_queries[0] 為改寫後的最終查詢問題
    """
    # ── 步驟 0：指代消解 ──────────────────────────────
    # 將 "那三國殺呢？" 這類問題利用歷史改寫成獨立問題
    resolved_question = rewrite_query_with_history(question, chat_history or [])
    if resolved_question != question:
        logger.info(f"DeepRAG 開始 (改寫後): 「{resolved_question}」")
    else:
        logger.info(f"DeepRAG 開始: 「{question}」")

    # ── 步驟 1：初始混合檢索 ──────────────────────────
    initial_docs = hybrid_search(resolved_question, vector_store, bm25_retriever)
    init_ctx = "\n\n".join([d.page_content for d in initial_docs]) if initial_docs else ""

    all_contexts = [init_ctx] if init_ctx.strip() else []
    all_queries = [resolved_question]
    seen_ctx = {init_ctx.strip()} if init_ctx.strip() else set()

    # 累積所有檢索到的候選文件（去重），供步驟 3 Rerank 用
    candidate_docs: list[Document] = []
    seen_doc_texts: set[str] = set()

    def _collect_candidates(docs: list[Document]) -> None:
        for d in docs:
            t = d.page_content.strip()
            if t and t not in seen_doc_texts:
                seen_doc_texts.add(t)
                candidate_docs.append(d)

    _collect_candidates(initial_docs)

    # ── 步驟 1b：粵語 BM25 展開搜尋 ──────────────────
    # 如果原始問題含粵語，用術語對照表產生書面語 queries 補充搜尋
    cantonese_expansions = expand_cantonese_terms(question)
    for exp_q in cantonese_expansions:
        if exp_q not in all_queries:
            all_queries.append(exp_q)
            exp_docs = hybrid_search(exp_q, vector_store, bm25_retriever)
            _collect_candidates(exp_docs)

    # ── 步驟 2：多輪迭代（充足性判斷 → follow-up） ──
    for rnd in range(DEEPRAG_MAX_ROUNDS):
        merged = "\n\n---\n\n".join(all_contexts).strip()
        if is_context_sufficient(resolved_question, merged):
            logger.info(f"第 {rnd+1} 輪: 充足，停止")
            break
        followups = generate_followup_queries(
            resolved_question, merged, chat_history=chat_history
        )
        if not followups:
            break
        new_found = False
        for q in followups:
            if q in all_queries:
                continue
            all_queries.append(q)
            new_docs = hybrid_search(q, vector_store, bm25_retriever)
            _collect_candidates(new_docs)
            new_ctx = "\n\n".join([d.page_content for d in new_docs])
            if not new_ctx.strip() or new_ctx.strip() in seen_ctx:
                continue
            seen_ctx.add(new_ctx.strip())
            all_contexts.append(new_ctx)
            new_found = True
        if not new_found:
            break

    # ── 步驟 3：Rerank（直接複用已收集的候選文件）───
    reranked = llm_rerank(resolved_question, candidate_docs)
    if reranked:
        final = "\n\n---\n\n".join([d.page_content for d in reranked]).strip()
    else:
        final = "\n\n---\n\n".join([c for c in all_contexts if c.strip()]).strip()

    logger.info(f"DeepRAG 完成: {len(all_queries)} queries, {len(reranked)} 段")
    return final, all_queries, reranked

