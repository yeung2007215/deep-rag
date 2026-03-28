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
)

logger = logging.getLogger(__name__)


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


def hybrid_search(
    query: str, vector_store: Chroma,
    bm25_docs: Optional[list[Document]] = None,
    k_vector: int = SIMILARITY_SEARCH_K,
    k_bm25: int = BM25_SEARCH_K,
    final_k: int = FINAL_TOP_K,
) -> list[Document]:
    """結合 BM25 + 向量語意檢索，去重合併"""
    try:
        vector_results = vector_store.similarity_search(query, k=k_vector)
    except Exception as e:
        logger.error(f"向量搜尋失敗: {e}")
        vector_results = []

    bm25_results = []
    if bm25_docs:
        try:
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = k_bm25
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
        f"你是檢索重排器。選出最能回答問題的段落（最多 {top_k} 段），"
        f"給 0-100 信心分數，只保留 ≥{thr} 的。\n"
        f"輸出格式（每行）：編號,信心\n不要輸出其他文字。"
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


def generate_followup_queries(question: str, context: str, n: int = DEEPRAG_FOLLOWUP_N) -> list[str]:
    """用 LLM 產生更精準的 follow-up 檢索 query"""
    prompt_sys = (
        f"你是檢索 query 生成器。為桌遊規則資料庫生成更精準的查詢。\n"
        f"輸出 {n} 條中文 query，每條一行，不加編號不解釋。\n"
        f"貼近規則書用語、嘗試同義詞、包含遊戲名稱。"
    )
    try:
        agent = create_agent(model=LLM_MODEL, system_prompt=prompt_sys)
        r = agent.invoke({"messages": [{"role": "user",
            "content": f"原始問題：{question}\n\n目前檢索內容：\n{context[:2000]}"}]})
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
    question: str, vector_store: Chroma,
    bm25_docs: Optional[list[Document]] = None,
) -> tuple[str, list[str], list[Document]]:
    """
    DeepRAG: 多輪迭代檢索 + 混合搜尋 + Rerank
    Returns: (final_context, all_queries, reranked_docs)
    """
    logger.info(f"DeepRAG 開始: 「{question}」")

    initial_docs = hybrid_search(question, vector_store, bm25_docs)
    init_ctx = "\n\n".join([d.page_content for d in initial_docs]) if initial_docs else ""

    all_contexts = [init_ctx] if init_ctx.strip() else []
    all_queries = [question]
    seen_ctx = {init_ctx.strip()} if init_ctx.strip() else set()

    for rnd in range(DEEPRAG_MAX_ROUNDS):
        merged = "\n\n---\n\n".join(all_contexts).strip()
        if is_context_sufficient(question, merged):
            logger.info(f"第 {rnd+1} 輪: 充足，停止")
            break
        followups = generate_followup_queries(question, merged)
        if not followups:
            break
        new_found = False
        for q in followups:
            if q in all_queries:
                continue
            all_queries.append(q)
            new_docs = hybrid_search(q, vector_store, bm25_docs)
            new_ctx = "\n\n".join([d.page_content for d in new_docs])
            if not new_ctx.strip() or new_ctx.strip() in seen_ctx:
                continue
            seen_ctx.add(new_ctx.strip())
            all_contexts.append(new_ctx)
            new_found = True
        if not new_found:
            break

    # Rerank all candidates
    cand_docs, seen_t = [], set()
    for q in all_queries:
        try:
            for d in vector_store.similarity_search(q, k=SIMILARITY_SEARCH_K):
                t = d.page_content.strip()
                if t and t not in seen_t:
                    seen_t.add(t)
                    cand_docs.append(d)
        except Exception:
            pass

    reranked = llm_rerank(question, cand_docs)
    if reranked:
        final = "\n\n---\n\n".join([d.page_content for d in reranked]).strip()
    else:
        final = "\n\n---\n\n".join([c for c in all_contexts if c.strip()]).strip()

    logger.info(f"DeepRAG 完成: {len(all_queries)} queries, {len(reranked)} 段")
    return final, all_queries, reranked

