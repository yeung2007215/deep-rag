"""
main.py — DeepRAG 桌遊規則問答系統入口
功能：
  - 兩層式交互選單（遊戲選擇 → 問題諮詢）
  - 多向量集合切換（Multi-collection）
  - 支援 back 返回上層 / quit 退出
  - 可選：執行文件索引 (--ingest)
"""

import logging
import sys
from typing import Optional

from langchain.agents import create_agent

from config import LLM_MODEL, GAME_COLLECTIONS, CHAT_HISTORY_MAX_TURNS, MAX_CONTEXT_CHARS
from retriever import (
    get_vector_store, get_all_documents, deep_rag_retrieve,
    build_bm25_retriever, ChatHistory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================
# 回答生成
# ==============================
def build_answer_agent(context: str, question: str, game_name: str):
    """根據檢索結果建立回答 agent（含 context 截斷保護）"""
    # 截斷保護：避免超出 LLM context window
    safe_context = context[:MAX_CONTEXT_CHARS]
    if len(context) > MAX_CONTEXT_CHARS:
        safe_context += f"\n…（以下省略，共 {len(context)} 字）"
        logger.warning(f"Context 過長，截斷至 {MAX_CONTEXT_CHARS} 字 (原 {len(context)} 字)")

    system_prompt = f"""你是《{game_name}》規則助理。以下是從規則資料庫檢索到的內容：

{safe_context}

請只根據以上內容回答：
問題：{question}

要求：
- 如可行，請引用你依據的規則句子（直接摘錄一小段即可）。
- 如果以上內容不足以回答，請回答：資料庫未找到相關規則，並指出你缺少哪個關鍵資訊。
"""
    return create_agent(model=LLM_MODEL, system_prompt=system_prompt)


def ask(
    question: str,
    vector_store,
    game_name: str,
    bm25_retriever=None,
    chat_history: Optional[ChatHistory] = None,
    verbose: bool = True,
) -> str:
    """完整的問答流程（智慧路由 Standard / DeepRAG）"""
    # 1. 智慧路由檢索（回傳第 4 個值 = 問題分類）
    final_context, all_queries, reranked_docs, query_type = deep_rag_retrieve(
        question, vector_store, bm25_retriever, chat_history=chat_history
    )

    if not final_context or not final_context.strip():
        return "❌ 資料庫中未找到相關規則資訊。請確認已執行文件索引 (python main.py --ingest)。"

    # 2. 顯示檢索資訊
    if verbose:
        # 路徑標籤
        path_labels = {
            "FACTOID": "⚡ Standard（事實查詢，不迭代）",
            "PROCEDURAL": "🔄 DeepRAG 輕量（流程步驟，1 輪迭代）",
            "REASONING": "🧠 DeepRAG 完整（推理分析，最多 3 輪）",
            "COMPARISON": "⚖️  DeepRAG 完整（跨規則比較，最多 3 輪）",
        }
        print("\n" + "=" * 60)
        print(f"🏷️  問題分類: {path_labels.get(query_type, query_type)}")
        # 指代消解顯示
        if all_queries and all_queries[0] != question:
            print(f"🔄 指代消解: 「{question}」")
            print(f"          → 「{all_queries[0]}」")
        history_len = len(chat_history) if chat_history else 0
        if history_len > 0:
            print(f"💬 對話歷史: 已記憶 {history_len} 輪 (最多保留 {CHAT_HISTORY_MAX_TURNS} 輪)")
        print(f"📋 使用的查詢 ({len(all_queries)} 條):")
        for i, q in enumerate(all_queries, 1):
            print(f"  [{i}] {q}")
        print(f"\n📊 Rerank 結果: {len(reranked_docs)} 段")
        if reranked_docs:
            for i, d in enumerate(reranked_docs, 1):
                gn = d.metadata.get("game_name", "?")
                print(f"  [{i}] ({gn}) {d.page_content[:80]}...")
        print("=" * 60)

    # 3. 生成回答
    try:
        agent = build_answer_agent(final_context, question, game_name)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"回答生成失敗: {e}")
        return f"❌ 回答生成時發生錯誤: {e}"


# ==============================
# 第一層：遊戲選擇選單
# ==============================
def show_game_menu() -> str | None:
    """顯示遊戲選單，回傳使用者選擇的 game_key 或 None（退出）"""
    game_keys = list(GAME_COLLECTIONS.keys())

    print("\n" + "=" * 60)
    print("🎲 DeepRAG Boardgame rule assistant")
    print("=" * 60)
    print("\nPlease select a game to query:\n")
    for i, key in enumerate(game_keys, 1):
        name = GAME_COLLECTIONS[key]["name"]
        print(f"  {i}. {name}")
    print(f"\n  Enter 'quit' or 'exit' to exit the program")
    print("-" * 40)

    while True:
        try:
            choice = input("\n🎮 Enter game No.: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice.lower() in ("quit", "exit", "q"):
            return None

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(game_keys):
                return game_keys[idx]

        print(f"  ⚠️  請輸入 1~{len(game_keys)} 的數字")


# ==============================
# 第二層：問題諮詢循環
# ==============================
def question_loop(game_key: str) -> bool:
    """
    針對指定遊戲的問答循環。
    Returns:
        True  — 使用者輸入 'back'，返回遊戲選擇
        False — 使用者輸入 'quit'/'exit'，結束程式
    """
    game_info = GAME_COLLECTIONS[game_key]
    game_name = game_info["name"]
    collection_name = game_info["collection"]

    # 連接對應的向量資料庫
    print(f"\n🔗 Connecting to {game_name} vector database ({collection_name})...")
    vector_store = get_vector_store(collection_name)

    # 建立 BM25 索引（只建一次，整個 session 複用）
    bm25_docs = get_all_documents(vector_store)
    bm25_retriever = None
    if not bm25_docs:
        logger.warning(f"{game_name} 的索引為空。請先執行 `python main.py --ingest`。")
    else:
        bm25_retriever = build_bm25_retriever(bm25_docs)
        print(f"   📚 BM25 索引: {len(bm25_docs)} 個文件")

    print("\n" + "=" * 60)
    print(f"📖 Current game: {game_name}")
    print("   Enter 'back' to return to game selection (chat history will be cleared)")
    print("   Enter 'quit' or 'exit' to exit the program")
    print("   Enter 'history' to view current chat history")
    print("=" * 60)

    # 對話歷史：每進入一個遊戲就重新開始累積
    chat_history: ChatHistory = []

    while True:
        try:
            question = input(f"\n❓ [{game_name}] Enter question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Good game！Bye!")
            return False

        if not question:
            continue

        lower_q = question.lower()

        if lower_q == "back":
            print(f"\n↩️  返回遊戲選擇... (已清除 {len(chat_history)} 輪對話記憶)")
            return True

        if lower_q in ("quit", "exit", "q"):
            print("\n👋 Good game！Bye!")
            return False

        if lower_q == "history":
            if not chat_history:
                print("  （目前沒有對話記憶）")
            else:
                print(f"\n💬 對話記憶（共 {len(chat_history)} 輪）：")
                for i, (q, a) in enumerate(chat_history, 1):
                    print(f"  [{i}] 問：{q}")
                    print(f"       答：{a[:120]}{'...' if len(a) > 120 else ''}")
            continue

        print("\n⏳ 正在檢索與分析中...\n")
        answer = ask(
            question, vector_store, game_name,
            bm25_retriever=bm25_retriever,
            chat_history=chat_history,
        )
        print(f"\n🤖 Answer:\n{answer}")

        # 將本輪問答加入歷史，超過上限時淘汰最舊的
        chat_history.append((question, answer))
        if len(chat_history) > CHAT_HISTORY_MAX_TURNS:
            chat_history.pop(0)


# ==============================
# 主程式入口
# ==============================
def main():
    # 處理 --ingest 命令
    if "--ingest" in sys.argv:
        from ingestion import ingest
        print("📥 Executing document indexing (all games)...")
        ingest(force="--force" in sys.argv)
        if "--interactive" not in sys.argv:
            print("\n✅ Indexing complete. Add --interactive to QA mode.")
            return

    # 處理 --query 單次查詢模式
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 >= len(sys.argv):
            print("❌ Please provide a question after --query")
            return
        question = sys.argv[idx + 1]

        # 從 --game 取得遊戲 key，預設第一個
        game_keys = list(GAME_COLLECTIONS.keys())
        game_key = game_keys[0]
        if "--game" in sys.argv:
            gi = sys.argv.index("--game")
            if gi + 1 < len(sys.argv):
                game_key = sys.argv[gi + 1]

        if game_key not in GAME_COLLECTIONS:
            print(f"❌ Unknown game: {game_key}")
            print(f"   Available: {', '.join(game_keys)}")
            return

        game_info = GAME_COLLECTIONS[game_key]
        vs = get_vector_store(game_info["collection"])
        bm25 = build_bm25_retriever(get_all_documents(vs))
        # CLI 單次查詢無對話歷史
        answer = ask(question, vs, game_info["name"], bm25_retriever=bm25, chat_history=None)
        print(f"\n🤖 Answer:\n{answer}")
        return

    # 巢狀循環：第一層（遊戲選擇）→ 第二層（問題諮詢）
    while True:
        game_key = show_game_menu()
        if game_key is None:
            print("\n👋 Good game！Bye!")
            break

        # 進入第二層問答；回傳 True 表示 back，False 表示 quit
        should_continue = question_loop(game_key)
        if not should_continue:
            break


if __name__ == "__main__":
    main()

