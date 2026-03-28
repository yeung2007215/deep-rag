from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Embedding model config
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Find ChromaDB collection, if not exist, create one
vector_store = Chroma(
    collection_name="asteriated_grail_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

# Tool for agent to retrieve information from ChromaDB
@tool
def get_rule(query: str) -> str:
    """Retrieve information"""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    results = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return results


# ==============================
# DeepRAG helper (新增：最少改動)
# ==============================

def _is_context_sufficient(context: str) -> bool:
    """
    很輕量的 sufficiency 檢查（你可按需要改）
    - 太短通常代表取唔到真正規則段落
    - 或者包含明顯「無資料」訊號
    """
    if not context:
        return False
    if len(context.strip()) < 120:
        return False
    # 你可加更多判斷，例如必須包含「起始/手牌/設置」之類關鍵字
    return True


def _generate_followup_queries(question: str, context: str, n: int = 3):
    """
    用 LLM 產生下一輪檢索 query（thinking-to-retrieve）
    注意：呢度用 deepseek chat 只做 query rewrite，唔要求工具呼叫。
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
原始問題：{question}

目前檢索內容：
{context}
"""
    rewrite_results = rewrite_agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    rewrite_messages = rewrite_results["messages"]
    text = rewrite_messages[-1].content.strip()

    # 每行一條 query
    queries = [line.strip() for line in text.splitlines() if line.strip()]
    return queries[:n] if queries else []


# Agent invoke
question = "星杯傳說, 起始手牌幾張"

# ==============================
# DeepRAG retrieval loop（核心改動：多輪檢索合併）
# ==============================

similarity_search_results = get_rule.invoke({"query": question})
all_contexts = []
all_queries = [question]
seen_chunks = set()

if similarity_search_results:
    all_contexts.append(similarity_search_results)

# 最多做 2 次「追加檢索」（你可改成 3 或更多）
max_rounds = 2

for _ in range(max_rounds):
    # 若已足夠就停止
    merged_context = "\n\n---\n\n".join(all_contexts).strip()
    if _is_context_sufficient(merged_context):
        break

    # 產生下一輪 query
    followups = _generate_followup_queries(question, merged_context, n=3)
    if not followups:
        break

    for q in followups:
        if q in all_queries:
            continue
        all_queries.append(q)

        new_context = get_rule.invoke({"query": q})
        if not new_context or not new_context.strip():
            continue

        # 去重（簡單用全文 hash；你也可按 chunk 粒度去重）
        key = new_context.strip()
        if key in seen_chunks:
            continue
        seen_chunks.add(key)

        all_contexts.append(new_context)

# 最終合併後的 context（DeepRAG 結果）
similarity_search_results = "\n\n---\n\n".join([c for c in all_contexts if c and c.strip()]).strip()


# Print("Test if retrieval success======")
if not similarity_search_results:
    print("No relevant information found in the database.")
else:
    print("================similarity_search_results (DeepRAG merged)================\n")
    print("retrieval success: Tool get_rule result:")
    print(similarity_search_results)

    print("\n================DeepRAG queries used================\n")
    for i, q in enumerate(all_queries, start=1):
        print(f"[{i}] {q}")


# Print AI reference (仍然保留你原本結構，但改為印最後一輪用到的 top-k)
print("================AI references================\n")
retrieved_docs = vector_store.similarity_search(question, k=3)
for index, reference in enumerate(retrieved_docs):
    print(f"Index of reference:{index}")
    print(reference)
    print(f"================End {index}================\n")


print("================RAG Answers================")
# Agent system prompt
system_prompt = f"""
    你是《星杯傳說》規則助理。以下是從規則資料庫檢索到的內容：
    {similarity_search_results}

    請只根據以上內容回答：
    問題：{question}

    要求：
    - 如可行，請引用你依據的規則句子（直接摘錄一小段即可）。
    - 如果以上內容不足以回答，請回答：資料庫未找到相關規則，並指出你缺少哪個關鍵資訊。
"""

# Create agent
agent = create_agent(
    model="deepseek:deepseek-chat",
    system_prompt=system_prompt,
    tools=[get_rule]  # 你這個版本其實已經手動 DeepRAG；保留 tools 亦無妨
)

# RAG Result
results = agent.invoke({"messages": [{"role": "user", "content": question}]})
messages = results["messages"]
print("\nAI answer:", messages[-1].content)