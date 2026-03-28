"""
main.py — DeepRAG 桌遊規則問答系統入口
功能：
  - 啟動時自動連接向量資料庫
  - 支援交互式終端問答循環
  - 整合 DeepRAG 多輪迭代檢索 + Rerank
  - 可選：執行文件索引 (--ingest)
"""

import logging
import sys

from langchain.agents import create_agent
from langchain.tools import tool

from config import LLM_MODEL
from retriever import get_vector_store, get_all_documents, deep_rag_retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_answer_agent(context: str, question: str):
    """根據檢索結果建立回答 agent"""
    system_prompt = f"""你是桌遊規則助理。以下是從規則資料庫檢索到的內容：

{context}

請只根據以上內容回答：
問題：{question}

要求：
- 如可行，請引用你依據的規則句子（直接摘錄一小段即可）。
- 如果以上內容不足以回答，請回答：資料庫未找到相關規則，並指出你缺少哪個關鍵資訊。
"""
    return create_agent(model=LLM_MODEL, system_prompt=system_prompt)


def ask(question: str, vector_store, bm25_docs=None, verbose: bool = True) -> str:
    """
    完整的 DeepRAG 問答流程
    Returns: AI 回答文字
    """
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
                game = d.metadata.get("game_name", "?")
                print(f"  [{i}] ({game}) {d.page_content[:80]}...")
        print("=" * 60)

    # 3. 生成回答
    try:
        agent = build_answer_agent(final_context, question)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"回答生成失敗: {e}")
        return f"❌ 回答生成時發生錯誤: {e}"


def interactive_loop(vector_store, bm25_docs=None):
    """交互式問答循環"""
    print("\n" + "=" * 60)
    print("🎲 DeepRAG 桌遊規則問答系統")
    print("   支援遊戲：星杯傳說、三國殺")
    print("   輸入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ 請輸入問題: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再見！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 再見！")
            break

        print("\n⏳ 正在檢索與分析中...\n")
        answer = ask(question, vector_store, bm25_docs)
        print(f"\n🤖 回答:\n{answer}")


def main():
    # 處理命令列參數
    if "--ingest" in sys.argv:
        from ingestion import ingest
        print("📥 執行文件索引...")
        ingest(force="--force" in sys.argv)
        if "--interactive" not in sys.argv:
            print("✅ 索引完成。加上 --interactive 可直接進入問答模式。")
            return

    # 連接向量資料庫
    print("🔗 連接向量資料庫...")
    vector_store = get_vector_store()

    # 建立 BM25 索引
    print("📚 建立 BM25 索引...")
    bm25_docs = get_all_documents(vector_store)
    if not bm25_docs:
        logger.warning("BM25 索引為空，將僅使用向量搜尋。執行 `python main.py --ingest` 先索引文件。")
        bm25_docs = None
    else:
        print(f"   BM25 索引包含 {len(bm25_docs)} 個文件")

    # 單次查詢模式
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 < len(sys.argv):
            question = sys.argv[idx + 1]
            answer = ask(question, vector_store, bm25_docs)
            print(f"\n🤖 回答:\n{answer}")
            return
        else:
            print("❌ 請在 --query 後面提供問題")
            return

    # 交互式模式
    interactive_loop(vector_store, bm25_docs)


if __name__ == "__main__":
    main()

