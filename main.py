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

from langchain.agents import create_agent

from config import LLM_MODEL, GAME_COLLECTIONS
from retriever import get_vector_store, get_all_documents, deep_rag_retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================
# 回答生成
# ==============================
def build_answer_agent(context: str, question: str, game_name: str):
    """根據檢索結果建立回答 agent"""
    system_prompt = f"""你是《{game_name}》規則助理。以下是從規則資料庫檢索到的內容：

{context}

請只根據以上內容回答：
問題：{question}

要求：
- 如可行，請引用你依據的規則句子（直接摘錄一小段即可）。
- 如果以上內容不足以回答，請回答：資料庫未找到相關規則，並指出你缺少哪個關鍵資訊。
"""
    return create_agent(model=LLM_MODEL, system_prompt=system_prompt)


def ask(question: str, vector_store, game_name: str,
        bm25_docs=None, verbose: bool = True) -> str:
    """完整的 DeepRAG 問答流程"""
    # 1. DeepRAG 檢索
    final_context, all_queries, reranked_docs = deep_rag_retrieve(
        question, vector_store, bm25_docs
    )

    if not final_context or not final_context.strip():
        return "❌ 資料庫中未找到相關規則資訊。請確認已執行文件索引 (python main.py --ingest)。"

    # 2. 顯示檢索資訊
    if verbose:
        print("\n" + "=" * 60)
        print("📋 DeepRAG 使用的查詢:")
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
    print("🎲 DeepRAG 桌遊規則問答系統")
    print("=" * 60)
    print("\n請選擇要查詢的遊戲：\n")
    for i, key in enumerate(game_keys, 1):
        name = GAME_COLLECTIONS[key]["name"]
        print(f"  {i}. {name}")
    print(f"\n  輸入 'quit' 或 'exit' 退出程式")
    print("-" * 40)

    while True:
        try:
            choice = input("\n🎮 請輸入編號: ").strip()
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
    print(f"\n🔗 連接 {game_name} 向量資料庫 ({collection_name})...")
    vector_store = get_vector_store(collection_name)

    # 建立 BM25 索引
    bm25_docs = get_all_documents(vector_store)
    if not bm25_docs:
        logger.warning(f"{game_name} 的索引為空。請先執行 `python main.py --ingest`。")
        bm25_docs = None
    else:
        print(f"   📚 BM25 索引: {len(bm25_docs)} 個文件")

    print("\n" + "=" * 60)
    print(f"📖 目前遊戲: {game_name}")
    print("   輸入 'back' 返回遊戲選擇")
    print("   輸入 'quit' 或 'exit' 退出程式")
    print("=" * 60)

    while True:
        try:
            question = input(f"\n❓ [{game_name}] 請輸入問題: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再見！")
            return False

        if not question:
            continue

        lower_q = question.lower()
        if lower_q == "back":
            print(f"\n↩️  返回遊戲選擇...")
            return True
        if lower_q in ("quit", "exit", "q"):
            print("\n👋 再見！")
            return False

        print("\n⏳ 正在檢索與分析中...\n")
        answer = ask(question, vector_store, game_name, bm25_docs)
        print(f"\n🤖 回答:\n{answer}")


# ==============================
# 主程式入口
# ==============================
def main():
    # 處理 --ingest 命令
    if "--ingest" in sys.argv:
        from ingestion import ingest
        print("📥 執行文件索引（所有遊戲）...")
        ingest(force="--force" in sys.argv)
        if "--interactive" not in sys.argv:
            print("\n✅ 索引完成。加上 --interactive 可直接進入問答模式。")
            return

    # 處理 --query 單次查詢模式
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 >= len(sys.argv):
            print("❌ 請在 --query 後面提供問題")
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
            print(f"❌ 未知遊戲: {game_key}")
            print(f"   可選: {', '.join(game_keys)}")
            return

        game_info = GAME_COLLECTIONS[game_key]
        vs = get_vector_store(game_info["collection"])
        bm25 = get_all_documents(vs) or None
        answer = ask(question, vs, game_info["name"], bm25)
        print(f"\n🤖 回答:\n{answer}")
        return

    # 巢狀循環：第一層（遊戲選擇）→ 第二層（問題諮詢）
    while True:
        game_key = show_game_menu()
        if game_key is None:
            print("\n👋 再見！")
            break

        # 進入第二層問答；回傳 True 表示 back，False 表示 quit
        should_continue = question_loop(game_key)
        if not should_continue:
            break


if __name__ == "__main__":
    main()

