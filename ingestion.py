"""
ingestion.py — 文件讀取、切分與向量化模組
功能：
  - 自動掃描 docs/ 下所有 .md 文件
  - 文件切分（RecursiveCharacterTextSplitter）
  - 去重機制（基於內容 hash，避免重複插入）
  - 儲存至 ChromaDB
"""

import glob
import hashlib
import logging
import os

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCS_DIR,
    CHROMA_PERSIST_DIR,
    GAME_COLLECTIONS,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """產生文本內容的 SHA-256 hash，用於去重"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_markdown_files(docs_dir: str = DOCS_DIR) -> list[str]:
    """掃描指定資料夾下的所有 Markdown 文件"""
    patterns = [os.path.join(docs_dir, "*.md"), os.path.join(docs_dir, "**/*.md")]
    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    files = sorted(files)
    logger.info(f"掃描到 {len(files)} 個 Markdown 文件: {[os.path.basename(f) for f in files]}")
    return files


def load_and_split_documents(file_paths: list[str]) -> list:
    """
    讀取多個 Markdown 文件並切分為 chunks
    每個 chunk 的 metadata 會包含來源檔名
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=CHUNK_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    all_splits = []
    for file_path in file_paths:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)

            # 為每個 chunk 加入來源檔名 metadata
            source_name = os.path.basename(file_path).replace("_規則說明書.md", "")
            for split in splits:
                split.metadata["game_name"] = source_name
                split.metadata["source_file"] = os.path.basename(file_path)
                split.metadata["content_hash"] = _content_hash(split.page_content)

            all_splits.extend(splits)
            logger.info(f"  ✅ {os.path.basename(file_path)}: 切分為 {len(splits)} 個 chunks")

        except Exception as e:
            logger.error(f"  ❌ 載入 {file_path} 失敗: {e}")
            continue

    logger.info(f"總計 {len(all_splits)} 個 chunks")
    return all_splits


def get_existing_hashes(vector_store: Chroma) -> set[str]:
    """取得 ChromaDB 中已存在的所有 content_hash，用於去重"""
    try:
        collection = vector_store._collection
        existing = collection.get(include=["metadatas"])
        hashes = set()
        if existing and existing.get("metadatas"):
            for meta in existing["metadatas"]:
                if meta and "content_hash" in meta:
                    hashes.add(meta["content_hash"])
        logger.info(f"ChromaDB 中已有 {len(hashes)} 個 chunks")
        return hashes
    except Exception as e:
        logger.warning(f"無法讀取既有 hash，將全部重新插入: {e}")
        return set()


def create_vector_store(collection_name: str) -> Chroma:
    """建立或連接指定名稱的 ChromaDB 向量資料庫"""
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _match_game_key(file_path: str) -> str | None:
    """根據檔名關鍵字匹配對應的遊戲 key，無法匹配則回傳 None"""
    basename = os.path.basename(file_path)
    for game_key, game_info in GAME_COLLECTIONS.items():
        for keyword in game_info["file_keywords"]:
            if keyword in basename:
                return game_key
    return None


def _ingest_to_collection(
    collection_name: str, splits: list, force: bool = False
) -> None:
    """將 chunks 寫入指定的 collection（含去重）"""
    if not splits:
        return

    vector_store = create_vector_store(collection_name)

    if not force:
        existing_hashes = get_existing_hashes(vector_store)
        new_splits = [
            s for s in splits
            if s.metadata.get("content_hash") not in existing_hashes
        ]
        skipped = len(splits) - len(new_splits)
        if skipped > 0:
            logger.info(f"  跳過 {skipped} 個已存在的 chunks")
    else:
        new_splits = splits

    if new_splits:
        ids = vector_store.add_documents(documents=new_splits)
        logger.info(f"  ✅ 寫入 {len(ids)} 個新 chunks → {collection_name}")
    else:
        logger.info(f"  無新 chunks 需要寫入 → {collection_name}")


def ingest(docs_dir: str = DOCS_DIR, force: bool = False) -> None:
    """
    主要入口：掃描文件 → 切分 → 按遊戲歸類 → 去重 → 寫入各自的 Collection
    """
    logger.info("=" * 50)
    logger.info("開始文件索引流程 (ingestion)")
    logger.info("=" * 50)

    # 1. 掃描文件
    file_paths = scan_markdown_files(docs_dir)
    if not file_paths:
        logger.warning(f"在 {docs_dir} 下未找到任何 .md 文件")
        return

    # 2. 讀取與切分
    all_splits = load_and_split_documents(file_paths)
    if not all_splits:
        logger.warning("所有文件切分後無內容")
        return

    # 3. 按遊戲歸類到不同 collection
    collection_splits: dict[str, list] = {}
    unmatched = []

    for split in all_splits:
        source_file = split.metadata.get("source_file", "")
        game_key = _match_game_key(source_file)
        if game_key:
            col_name = GAME_COLLECTIONS[game_key]["collection"]
            collection_splits.setdefault(col_name, []).append(split)
        else:
            unmatched.append(split)

    if unmatched:
        logger.warning(f"有 {len(unmatched)} 個 chunks 無法匹配遊戲，將跳過")

    # 4. 分別寫入各 collection
    for col_name, splits in collection_splits.items():
        logger.info(f"\n📦 索引至集合: {col_name} ({len(splits)} chunks)")
        _ingest_to_collection(col_name, splits, force=force)

    logger.info("\n✅ 所有集合索引流程完成")


if __name__ == "__main__":
    import sys
    force_mode = "--force" in sys.argv
    ingest(force=force_mode)

