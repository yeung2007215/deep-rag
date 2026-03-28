from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Embedding / dynamic vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
current_vector_store = None

# Game collection mapping
GAME_COLLECTIONS = {
    "1": ("asteriated_grail_collection", "星杯傳說"),
    "2": ("war_of_the_three_kingdom_collection", "三國殺")
}


@tool
def get_rule(query: str) -> str:
    """從選定的桌遊規則資料庫中檢索資訊。"""
    if current_vector_store is None:
        return "尚未選擇遊戲，無法檢索。"

    retrieved_docs = current_vector_store.similarity_search(query, k=20)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


def _is_context_sufficient(context: str) -> bool:
    if not context:
        return False
    return len(context.strip()) >= 120


def _generate_followup_queries(user_question: str, context: str, game_name: str, n: int = 3):
    rewrite_system_prompt = f"""
你是檢索 query 生成器。目標：為《{game_name}》規則資料庫生成更精準的檢索查詢。
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

    idxs = [int(part) for part in text.split(",") if part.isdigit()]
    seen = set()
    picked = []
    for i in idxs:
        if 1 <= i <= len(docs) and i not in seen:
            picked.append(i)
            seen.add(i)
        if len(picked) >= top_k:
            break

    return [docs[i - 1] for i in picked]


def select_collection(choice: str):
    if choice == "1" or "星杯" in choice:
        return GAME_COLLECTIONS["1"]
    if choice == "2" or "三國" in choice:
        return GAME_COLLECTIONS["2"]
    return None, None


def init_vector_store(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="./chroma_langchain_db"
    )


print("=== 🦘 Welcome to BoardgameRoo DeepRAG ===")

while True:
    print("\n🕹️ Game list:")
    print("[1] ⭐️ 星杯傳說 (asteriated_grail_collection)")
    print("[2] ⚔️ 三國殺 (war_of_the_three_kingdom_collection)")
    print("Input 'exit' to exit the program")

    choice = input("\n🦘: Enter game number or name (1/2): ").strip()
    if choice.lower() in ["exit", "quit", "退出"]:
        break

    target_collection, game_name = select_collection(choice)
    if not target_collection:
        print("[Error] Invalid choice, please retry.")
        continue

    current_vector_store = init_vector_store(target_collection)
    print(f"\n🦘 You are in [{game_name}] rules mode.")
    print("(Enter 'back' to return to game selection, 'exit' to quit program)")

    while True:
        question = input(f"\n[{game_name}] Your question: ").strip()
        if not question:
            continue
        if question.lower() == "back":
            print("Back to game list...")
            break
        if question.lower() in ["exit", "quit", "退出"]:
            print("Programme exiting...")
            exit()

        similarity_search_results = get_rule.invoke({"query": question})
        all_contexts = []
        all_queries = [question]
        seen_contexts = set()

        if similarity_search_results and similarity_search_results.strip():
            all_contexts.append(similarity_search_results)
            seen_contexts.add(similarity_search_results.strip())

        max_rounds = 20
        for _ in range(max_rounds):
            merged_context = "\n\n---\n\n".join(all_contexts).strip()
            if _is_context_sufficient(merged_context):
                break

            followups = _generate_followup_queries(question, merged_context, game_name, n=3)
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

        merged_context = "\n\n---\n\n".join([c for c in all_contexts if c and c.strip()]).strip()
        candidate_docs_all = []
        seen_text = set()

        for q in all_queries:
            docs = current_vector_store.similarity_search(q, k=10)
            for d in docs:
                t = d.page_content.strip()
                if not t or t in seen_text:
                    continue
                seen_text.add(t)
                candidate_docs_all.append(d)

        reranked_docs = deepseek_rerank_docs(question, candidate_docs_all, top_k=5)
        if reranked_docs:
            similarity_search_results = "\n\n---\n\n".join([d.page_content for d in reranked_docs]).strip()
        else:
            similarity_search_results = merged_context

        if not similarity_search_results or not similarity_search_results.strip():
            print("警告：在資料庫中找不到相關片段，AI 將嘗試以現有知識回答。")

        system_prompt = f"""
你是《{game_name}》的桌遊規則助理。
以下是從該遊戲規則資料庫檢索到的相關內容：
{similarity_search_results}

請只根據以上內容回答：
問題：{question}

要求：
- 如可行，請引用你依據的規則句子（直接摘錄一小段即可）。
- 如果以上內容不足以回答，請回答：在目前的規則書中找不到相關說明。
"""

        agent = create_agent(
            model="deepseek:deepseek-chat",
            system_prompt=system_prompt,
            tools=[get_rule]
        )

        try:
            print("🤖 I am thinking...")
            results = agent.invoke({"messages": [{"role": "user", "content": question}]})
            ai_answer = results["messages"][-1].content

            print("\n========= AI Answer =========")
            print(ai_answer)
            print("========= End =========\n")

            

            
            # ==============================================================
            # Only for develpr testing: print retrieved chunks for reference
            retrieved_docs = current_vector_store.similarity_search(question, k=10)
            print(f"\n[AI reference]: Found {len(retrieved_docs)} chunks")
            for index, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get('source', 'Unknown source') if hasattr(doc, 'metadata') else 'Unknown source'
                print(f"=====[AI reference: Chunk no.{index}]=====")
                print(f"Source: {source}")
                print(f"Content: {doc.page_content[:500]}...")
                print("[End of chunk]\n")

        except Exception as e:
            print(f"Error occurred: {e}")
