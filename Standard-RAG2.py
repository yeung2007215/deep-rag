from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

# Get docuemnt from markdown file and load into ChromaDB
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ✅ Hybrid Search (BM25 + Vector)
from rank_bm25 import BM25Okapi
import re


# =========================
# Embedding model config
# =========================
embedding = OllamaEmbeddings(model="nomic-embed-text")
current_vector_store = None

# ✅ BM25 globals
current_bm25 = None
current_bm25_docs = None   # list[str]
current_bm25_meta = None   # list[dict]


# =========================
# Hybrid Search helpers
# =========================
def _tokenize(text: str):
    """
    簡易 tokenizer：
    - 英文/數字：以 word token 為主
    - 中文：用 2-gram 提升匹配（比逐字更穩）
    """
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()

    # 偏英文 / 數字：用 word tokens
    if re.search(r"[a-z0-9]", text):
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens if tokens else list(text)

    # 中文：2-gram
    text = re.sub(r"[^\u4e00-\u9fff0-9a-z]+", "", text)
    if len(text) <= 2:
        return list(text)
    return [text[i:i+2] for i in range(len(text) - 1)]


def build_bm25_index_from_chroma(store: Chroma):
    """
    從 Chroma collection 拉出全部 documents，建立 BM25 index。
    """
    data = store.get(include=["documents", "metadatas"])
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or [{} for _ in docs]

    tokenized_corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs, metas


def hybrid_candidates(query: str, bm25_k: int = 30, vec_k: int = 30, alpha: float = 0.55):
    """
    產生候選池（Documents），用於 rerank
    - BM25：用 docs 文本打分，回傳 doc 文本 + 分數
    - Vector：用 Chroma similarity_search_with_score，回傳 Document + 分數
    最後合併成「唯一候選文本」清單（以 page_content 去重）
    """
    global current_vector_store, current_bm25, current_bm25_docs

    if current_vector_store is None:
        return []

    # ---------- BM25 top ----------
    bm25_texts = []
    if current_bm25 is not None and current_bm25_docs:
        q_tokens = _tokenize(query)
        scores = current_bm25.get_scores(q_tokens)

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_k]
        for i in top_idx:
            t = (current_bm25_docs[i] or "").strip()
            if t:
                bm25_texts.append(t)

    # ---------- Vector top ----------
    vec_docs = []
    try:
        docs_scores = current_vector_store.similarity_search_with_score(query, k=vec_k)
        for d, _dist in docs_scores:
            if d and (d.page_content or "").strip():
                vec_docs.append(d)
    except Exception:
        docs = current_vector_store.similarity_search(query, k=vec_k)
        for d in docs:
            if d and (d.page_content or "").strip():
                vec_docs.append(d)

    # ---------- Merge candidates (dedup by text) ----------
    seen = set()
    candidates = []

    # 先放向量候選（通常語意相關度較穩）
    for d in vec_docs:
        t = d.page_content.strip()
        if t not in seen:
            seen.add(t)
            candidates.append(d)

    # 再把 BM25 候選補進去（BM25 只得 text，轉成 Document-like）
    # 這裡用最簡方式：做一個最小 object，保持 page_content 屬性
    class _Doc:
        def __init__(self, page_content):
            self.page_content = page_content
            self.metadata = {"source": "BM25"}

    for t in bm25_texts:
        if t not in seen:
            seen.add(t)
            candidates.append(_Doc(t))

    return candidates


# =========================
# DeepSeek rerank (align with DeepRAG search method, but single-shot)
# =========================
def deepseek_rerank_docs(user_question: str, docs, top_k: int = 8):
    """
    使用 DeepSeek 對候選 docs 做 rerank（單次，不做 DeepRAG 多輪）
    回傳 top_k docs
    """
    if not docs:
        return []

    # 壓縮候選段落，避免 prompt 太長
    blocks = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").replace("\n", " ").strip()
        if not text:
            continue
        blocks.append(f"[{i}] {text[:450]}")
    if not blocks:
        return []

    candidates_text = "\n".join(blocks)

    rerank_system_prompt = f"""
你是一個「檢索重排器 (reranker)」。給你一條桌遊規則問題，以及多段候選規則文本（已編號）。
請選出最能直接回答問題的 {top_k} 段。

輸出格式要求：
- 只輸出編號，用逗號分隔，例如：3,1,7,2,5
- 不要解釋，不要輸出多餘文字
"""

    rerank_agent = create_agent(
        model="deepseek:deepseek-chat",
        system_prompt=rerank_system_prompt,
    )

    prompt = f"問題：{user_question}\n\n候選段落：\n{candidates_text}"
    r = rerank_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    text = r["messages"][-1].content.strip().replace(" ", "")

    idxs = []
    for part in text.split(","):
        if part.isdigit():
            idxs.append(int(part))

    # 去重 + 範圍檢查 + 截取 top_k
    seen = set()
    picked = []
    for i in idxs:
        if 1 <= i <= len(docs) and i not in seen:
            picked.append(i)
            seen.add(i)
        if len(picked) >= top_k:
            break

    return [docs[i - 1] for i in picked]


def build_context_from_reranked(docs, max_chunks: int = 8):
    """
    將 reranked docs 組成 context（RAG context）
    """
    if not docs:
        return ""
    chosen = docs[:max_chunks]
    return "\n\n---\n\n".join([(d.page_content or "").strip() for d in chosen if (d.page_content or "").strip()]).strip()


# =========================
# Tool for agent retrieval (Aligned Search: Hybrid + DeepSeek rerank, but no DeepRAG loop)
# =========================
@tool
def get_rule(query: str) -> str:
    """從選定的桌遊規則資料庫中檢索資訊（Aligned: Hybrid candidates + DeepSeek rerank）。"""
    if current_vector_store is None:
        return "尚未選擇遊戲，無法檢索。"

    # 1) Hybrid candidates (BM25 + Vector)
    candidates = hybrid_candidates(query, bm25_k=30, vec_k=30, alpha=0.55)

    # 2) DeepSeek rerank (single-shot)
    reranked_docs = deepseek_rerank_docs(query, candidates, top_k=8)

    # 3) Build final context
    return build_context_from_reranked(reranked_docs, max_chunks=8)


print("=== 🦘  Welcome to use BoardgameRoo 🦘 ===")

# 1st loop：select collection
while True:
    print("\n🕹️ Game list:")
    print("[1] ⭐️ 星杯傳說 (asteriated_grail_collection)")
    print("[2] ⚔️  三國殺 (war_of_the_three_kingdom_collection)")
    print("Input 'exit' to exit the program")

    choice = input("\n🦘: Enter game number or name (1/2): ").strip()

    if choice.lower() in ['exit', 'quit', '退出']:
        break

    # Selecting collection by user input
    if choice == "1" or "星杯" in choice:
        target_collection = "asteriated_grail_collection"
        game_name = "星杯傳說"
    elif choice == "2" or "三國殺" in choice:
        target_collection = "war_of_the_three_kingdom_collection"
        game_name = "三國殺"
    else:
        print("[Error] Invalid choice, please retry.")
        continue

    # Init Vector Store
    current_vector_store = Chroma(
        collection_name=target_collection,
        embedding_function=embedding,
        persist_directory="./chroma_langchain_db"
    )

    # ✅ Build BM25 index once per selected game
    print("Building BM25 index for hybrid search...")
    current_bm25, current_bm25_docs, current_bm25_meta = build_bm25_index_from_chroma(current_vector_store)
    print(f"BM25 ready. docs={len(current_bm25_docs)}")

    print(f"\n🦘 You are in[{game_name}]")
    print(f"(Enter'back' to return to game selection, 'exit' to quit program)")

    # 2nd loop：focus on the selected collection, ask questions
    while True:
        question = input(f"\n[{game_name}] Your question: ").strip()

        if not question:
            continue

        if question.lower() == 'back':
            print("Back to game list...")
            break

        if question.lower() in ['exit', 'quit', '退出']:
            print("Programme exiting...")
            exit()

        # Execute aligned search (Hybrid + DeepSeek rerank)
        similarity_search_results = get_rule.invoke({"query": question})

        if not similarity_search_results.strip():
            print("警告：在資料庫中找不到相關片段，AI 將嘗試以現有知識回答。")

        # Set system prompt
        system_prompt = f"""
你是《{game_name}》的桌遊規則助理。
以下是從該遊戲規則資料庫檢索到的相關內容：
{similarity_search_results}

指令：
1. 請優先根據以上提供的檢索內容回答問題。
2. 如果檢索內容中完全沒有提到相關資訊，請老實回答「在目前的規則書中找不到相關說明」。
3. 回答必須準確且語氣專業。
"""

        # Create Agent
        agent = create_agent(
            model="deepseek:deepseek-chat",
            system_prompt=system_prompt,
            tools=[get_rule]
        )

        try:
            print("🤖I am thinking...")
            results = agent.invoke({"messages": [{"role": "user", "content": question}]})
            ai_answer = results["messages"][-1].content

            print("\n========= AI Answer =========")
            print(ai_answer)
            print("========= End =========\n")
            
            print("Reference:")

            
            # print("Reference:")
            # for index, i in enumerate(similarity_search_results.split("\n\n---\n\n"), 1):
            #     print(f"[{index}] {i}")

        except Exception as e:
            print(f"Error occurred: {e}")