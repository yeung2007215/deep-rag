from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ==============================
# Embedding / Vector DB
# ==============================
embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="asteriated_grail_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)


# ==============================
# Tool: retrieve from Chroma
# ==============================
@tool
def get_rule(query: str) -> str:
    """Retrieve information"""
    retrieved_docs = vector_store.similarity_search(query, k=20)
    results = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return results


# ==============================
# DeepRAG helpers
# ==============================
def _is_context_sufficient(context: str) -> bool:
    """
    Very lightweight sufficiency check.
    You can adjust thresholds as needed.
    """
    if not context:
        return False
    if len(context.strip()) < 120:
        return False
    return True


def _generate_followup_queries(user_question: str, context: str, n: int = 3):
    """
    Use LLM to generate additional retrieval queries (thinking-to-retrieve).
    Output: list of n queries, one per line.
    """
    rewrite_system_prompt = f"""
你是檢索 query 生成器。目標：為《星杯傳說》規則資料庫生成更精準的檢索查詢。
你會收到：
1) 原始問題
2) 目前檢索到的內容（可能不足）

請輸出 {n} 條「中文檢索 query」，每條一行，不要加編號，不要解釋。
要求：
- query 要更貼近規則書用語（例如：設置/初始/起手/抽牌/手牌上限/起始手牌）
- 可以嘗試同義詞與更具體關鍵字
- 不要輸出空行
"""

    rewrite_agent = create_agent(
        model="deepseek:deepseek-chat",
        system_prompt=rewrite_system_prompt
    )

    user_prompt = f"""
原始問題：{user_question}

目前檢索內容：
{context}
"""
    rewrite_results = rewrite_agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    text = rewrite_results["messages"][-1].content.strip()

    queries = [line.strip() for line in text.splitlines() if line.strip()]
    return queries[:n] if queries else []


def deepseek_rerank_docs(user_question: str, docs, top_k: int = 5):
    """
    Use DeepSeek as a reranker to select the most relevant chunks.
    Returns top_k documents (LangChain Document objects).
    """
    if not docs:
        return []

    blocks = []
    for i, d in enumerate(docs, start=1):
        text = d.page_content.replace("\n", " ").strip()
        blocks.append(f"[{i}] {text[:450]}")
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
        system_prompt=rerank_system_prompt
    )

    prompt = f"問題：{user_question}\n\n候選段落：\n{candidates_text}"
    r = rerank_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    text = r["messages"][-1].content.strip().replace(" ", "")

    idxs = []
    for part in text.split(","):
        if part.isdigit():
            idxs.append(int(part))

    seen = set()
    picked = []
    for i in idxs:
        if 1 <= i <= len(docs) and i not in seen:
            picked.append(i)
            seen.add(i)
        if len(picked) >= top_k:
            break

    return [docs[i - 1] for i in picked]


# ==============================
# Agent invoke
# ==============================
user_question = "星杯傳說, 起始手牌幾張"

# First retrieval
similarity_search_results = get_rule.invoke({"query": user_question})
all_contexts = []
all_queries = [user_question]
seen_contexts = set()

if similarity_search_results and similarity_search_results.strip():
    all_contexts.append(similarity_search_results)
    seen_contexts.add(similarity_search_results.strip())

# DeepRAG iterative retrieval
max_rounds = 20
for _ in range(max_rounds):
    merged_context = "\n\n---\n\n".join(all_contexts).strip()

    if _is_context_sufficient(merged_context):
        break

    followups = _generate_followup_queries(user_question, merged_context, n=3)
    if not followups:
        break

    for q in followups:
        if q in all_queries:
            continue
        all_queries.append(q)

        new_context = get_rule.invoke({"query": q})
        if not new_context or not new_context.strip():
            continue

        key = new_context.strip()
        if key in seen_contexts:
            continue
        seen_contexts.add(key)
        all_contexts.append(new_context)

# Merge contexts (fallback if rerank fails)
merged_context = "\n\n---\n\n".join([c for c in all_contexts if c and c.strip()]).strip()

# ==============================
# DeepSeek rerank (collect candidates using all_queries)
# ==============================
candidate_docs_all = []
seen_text = set()

for q in all_queries:
    docs = vector_store.similarity_search(q, k=10)
    for d in docs:
        t = d.page_content.strip()
        if not t:
            continue
        if t in seen_text:
            continue
        seen_text.add(t)
        candidate_docs_all.append(d)

reranked_docs = deepseek_rerank_docs(user_question, candidate_docs_all, top_k=5)

if reranked_docs:
    similarity_search_results = "\n\n---\n\n".join([d.page_content for d in reranked_docs]).strip()
else:
    similarity_search_results = merged_context


# ==============================
# Debug prints
# ==============================
if not similarity_search_results or not similarity_search_results.strip():
    print("No relevant information found in the database.")
else:
    print("================similarity_search_results (DeepRAG merged + DeepSeek rerank)================\n")
    print(similarity_search_results)

    print("\n================DeepRAG queries used================\n")
    for i, q in enumerate(all_queries, start=1):
        print(f"[{i}] {q}")

print("\n================DeepSeek rerank picked (top-k)================\n")
if reranked_docs:
    for i, d in enumerate(reranked_docs, start=1):
        print(f"[{i}]")
        print(d.page_content[:300])
        print("metadata:", getattr(d, "metadata", None))
        print("-----")
else:
    print("(rerank returned empty; using merged_context)")

print("\n================RAG Answers================\n")


# ==============================
# Agent system prompt
# ==============================
system_prompt = f"""
你是《星杯傳說》規則助理。以下是從規則資料庫檢索到的內容：
{similarity_search_results}

請只根據以上內容回答：
問題：{user_question}

要求：
- 如可行，請引用你依據的規則句子（直接摘錄一小段即可）。
- 如果以上內容不足以回答，請回答：資料庫未找到相關規則，並指出你缺少哪個關鍵資訊。
"""

# Create agent
agent = create_agent(
    model="deepseek:deepseek-chat",
    system_prompt=system_prompt,
    tools=[get_rule]  #DeepRAG + rerank
)

# RAG Result
results = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
messages = results["messages"]
print("\nAI answer:", messages[-1].content)