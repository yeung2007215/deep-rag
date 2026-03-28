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
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "boardgame_rules_collection")

# ==============================
# Embedding 模型設定
# ==============================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ==============================
# LLM 設定
# ==============================
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek:deepseek-chat")

# ==============================
# 文件切分參數
# ==============================
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
# Rerank 參數
# ==============================
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
# Rerank 信心門檻（0~1，低於此門檻的候選段落不納入）
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.3"))

