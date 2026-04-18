import logging
import sys
from langchain.agents import create_agent
from config import LLM_MODEL, GAME_COLLECTIONS, MAX_CONTEXT_CHARS
# 正式對接 retriever.py 的功能
from retriever import (
    get_vector_store, 
    get_all_documents, 
    build_bm25_retriever, 
    rewrite_query_with_history,
    hybrid_search
)

# 設定 Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def chat():
    print("=== 🦘 Welcome to BoardgameRoo (Standard RAG Mode) 🦘 ===")
    
    while True:
        print("\n🕹️ Game list:")
        for i, (key, info) in enumerate(GAME_COLLECTIONS.items(), 1):
            print(f"[{i}] {info['name']} ({info['collection']})")
        
        choice = input("\n🦘: Enter game number or 'exit': ").strip()
        if choice.lower() in ['exit', 'quit']: break

        # 根據輸入對應遊戲配置
        keys = list(GAME_COLLECTIONS.keys())
        try:
            selected_key = keys[int(choice) - 1]
            game_info = GAME_COLLECTIONS[selected_key]
        except (ValueError, IndexError):
            print("Invalid choice, please try again."); continue

        # 1. 連接向量資料庫 (透過 retriever.py)
        print(f"🔗 Connecting to {game_info['name']} database...")
        vector_store = get_vector_store(game_info["collection"])

        # 2. 建立 BM25 索引 (透過 retriever.py)
        print("📚 Building BM25 index...")
        all_docs = get_all_documents(vector_store)
        bm25_retriever = build_bm25_retriever(all_docs)

        chat_history = [] # 初始化對話歷史
        print(f"\n✅ Ready! Ask me anything about 《{game_info['name']}》.")

        while True:
            question = input(f"\n[{game_info['name']}] User: ").strip()
            if not question: continue
            if question.lower() == 'back': break

            # --- 步驟 A: 重寫查詢 (支援廣東話轉換與歷史關聯) ---
            # 這裡會自動處理 CANTONESE_TERM_MAP
            resolved_question = rewrite_query_with_history(question, chat_history)
            if resolved_question != question:
                print(f"🔍 Optimized Query: {resolved_question}")

            # --- 步驟 B: 單次 Hybrid Search (不執行多輪迭代) ---
            print("🤖 Searching rules...")
            # 直接使用 retriever 裡的 hybrid_search 函數，獲得 Vector + BM25 合併結果
            retrieved_docs = hybrid_search(resolved_question, vector_store, bm25_retriever)
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            
            # 截斷保護
            safe_context = context[:MAX_CONTEXT_CHARS]

            # --- 步驟 C: 生成回答 ---
            system_prompt = f"""你是《{game_info['name']}》規則助理。
以下是從規則庫檢索到的內容：
{safe_context}

請根據以上內容準確回答問題。如果內容未提及，請老實告知。"""
            
            agent = create_agent(model=LLM_MODEL, system_prompt=system_prompt)

            try:
                print("🧠 Thinking...")
                results = agent.invoke({"messages": [{"role": "user", "content": resolved_question}]})
                ai_answer = results["messages"][-1].content
                
                print(f"\n========= AI Answer =========\n{ai_answer}\n=============================")
                
                # 更新對話歷史
                chat_history.append((question, ai_answer))
                if len(chat_history) > 5: chat_history.pop(0)

            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat()