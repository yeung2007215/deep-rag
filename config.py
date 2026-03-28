"""
deep-rag 專案配置模組
統一管理所有模型、資料庫路徑、檢索參數等設定
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 路徑設定
# ==============================
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_langchain_db")

# ==============================
# 多遊戲集合定義 (Multi-collection)
# ==============================
# 每個遊戲對應一個獨立的 ChromaDB Collection
# key: 遊戲識別碼, value: dict(name=顯示名稱, collection=集合名稱, file_keywords=檔名關鍵字)
GAME_COLLECTIONS = {
    "asteriated_grail": {
        "name": "星杯傳說",
        "collection": "asteriated_grail_collection",
        "file_keywords": ["星杯傳說"],
    },
    "war_of_three_kingdoms": {
        "name": "三國殺",
        "collection": "war_of_the_three_kingdom_collection",
        "file_keywords": ["三國殺"],
    },
}

# 預設集合（向後相容）
DEFAULT_GAME_KEY = os.getenv("DEFAULT_GAME_KEY", "asteriated_grail")
CHROMA_COLLECTION_NAME = GAME_COLLECTIONS[DEFAULT_GAME_KEY]["collection"]

# ==============================
# Embedding 模型設定
# ==============================
# nomic-embed-text: 支援多語言但中文語義捕捉較弱
# 升級建議（按效果排序）：
#   - "bge-m3"       : BAAI 出品，中英文混合效果最佳，須 ollama pull bge-m3
#   - "bge-large-zh" : 純中文語義最強，ollama pull bge-large-zh（需要較多 RAM）
#   - "nomic-embed-text" : 輕量，適合開發測試
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ==============================
# LLM 設定
# ==============================
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek:deepseek-chat")

# ==============================
# 文件切分參數
# ==============================
# 桌遊規則說明書建議：
#   CHUNK_SIZE=300~500: 規則書段落通常短，太大會混入不相關規則
#   CHUNK_OVERLAP=50~100: 足夠保留上下文連貫性
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", " ", ""]

# ==============================
# 檢索參數
# ==============================
# 向量相似度搜尋 top-k
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "15"))
# BM25 搜尋 top-k
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "15"))
# 最終回傳 top-k（去重合併後）
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))

# ==============================
# DeepRAG 迭代參數
# ==============================
# 最大迭代輪數
DEEPRAG_MAX_ROUNDS = int(os.getenv("DEEPRAG_MAX_ROUNDS", "3"))
# 每輪產生的 follow-up query 數量
DEEPRAG_FOLLOWUP_N = int(os.getenv("DEEPRAG_FOLLOWUP_N", "3"))
# Context 充足性最低字數門檻
DEEPRAG_MIN_CONTEXT_LENGTH = int(os.getenv("DEEPRAG_MIN_CONTEXT_LENGTH", "120"))

# ==============================
# 對話記憶參數
# ==============================
# 傳入 LLM 的最近幾輪對話（太多會超出 context window）
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "5"))
# 每輪歷史回答的最大字元數（截斷保護，避免長回答爆 context window）
CHAT_HISTORY_ANSWER_MAX_CHARS = int(os.getenv("CHAT_HISTORY_ANSWER_MAX_CHARS", "300"))
# Context 傳入 LLM 的最大字元數（DeepSeek context window ≈ 64K tokens ≈ 32K 中文字）
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

# ==============================
# 廣東話 → 書面語 術語對照表
# ==============================
# 用於 BM25 query 展開：將粵語關鍵字替換/追加書面語同義詞
# 格式：粵語 → 書面語（可一對多）
CANTONESE_TERM_MAP: dict[str, list[str]] = {
    # 通用桌遊術語
    "點樣": ["如何", "怎麼"],
    "點先": ["怎樣才能", "如何才能"],
    "幾多": ["多少"],
    "幾張": ["多少張", "幾張"],
    "攞": ["拿", "取得", "獲得"],
    "擺": ["放置", "放"],
    "掟": ["丟棄", "棄置"],
    "揀": ["選擇", "挑選"],
    "贏": ["勝利", "獲勝"],
    "輸": ["落敗", "失敗"],
    "出牌": ["打出", "使用"],
    "派牌": ["發牌", "分配手牌"],
    "洗牌": ["洗牌", "重新洗混"],
    "摸牌": ["抽牌", "摸牌"],
    "手牌": ["手牌", "手中的牌"],
    "回合": ["回合", "輪次"],
    "嘅": ["的"],
    "佢": ["他", "她", "它"],
    "咗": ["了"],
    "唔": ["不", "沒有"],
    "冇": ["沒有", "無"],
    "係": ["是"],
    "同": ["和", "與"],
    "嗰個": ["那個"],
    "嗰啲": ["那些"],
    "呢個": ["這個"],
    "乜嘢": ["什麼"],
    "邊個": ["誰", "哪個"],
    "幾時": ["什麼時候", "何時"],
}

# ==============================
# Rerank 參數
# ==============================
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
# Rerank 信心門檻（0~1，低於此門檻的候選段落不納入）
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.3"))

