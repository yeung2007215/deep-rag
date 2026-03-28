import re
import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_community.retrievers import BM25Retriever

# 1. 讀取原始檔
loader = TextLoader("docs/星杯傳說_規則說明書.md", encoding="utf-8")
raw_docs = loader.load()

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 稍微加大一點，確保註解跟內容在一起
    chunk_overlap=100
)
initial_splits = text_splitter.split_documents(raw_docs)

# 【核心修正】更穩健的 Metadata 提取器
def process_metadata(docs: List[Document]) -> List[Document]:
    processed = []
    for doc in docs:
        content = doc.page_content
        new_metadata = doc.metadata.copy()
        
        # 使用正則搜尋註解區塊
        # 改用更寬鬆的匹配，並檢查是否有抓到內容
        meta_match = re.search(r"", content, re.DOTALL)
        
        if meta_match:
            try:
                # 確保 group(1) 存在
                meta_text = meta_match.group(1)
                
                # 分行解析 key: value
                lines = meta_text.split('\n')
                for line in lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        key = key.strip()
                        val = val.strip().strip('- ') # 移除 YAML 的列表符號
                        
                        if key == "level":
                            try:
                                new_metadata[key] = int(val)
                            except:
                                new_metadata[key] = 5
                        else:
                            new_metadata[key] = val
            except IndexError:
                pass # 如果沒抓到 group 則跳過
        
        # 移除文本中的註解，保持內容純淨
        clean_content = re.sub(r"", "", content, flags=re.DOTALL).strip()
        
        # 如果沒抓到 level，給予預設值 5 (內文層級)
        if "level" not in new_metadata:
            new_metadata["level"] = 5
        if "path" not in new_metadata:
            new_metadata["path"] = "未知路徑"
            
        processed.append(Document(page_content=clean_content, metadata=new_metadata))
    return processed

# 執行提取
all_docs = process_metadata(initial_splits)

# 3. 建立向量庫 (Chroma)
embedding = OllamaEmbeddings(model="nomic-embed-text")
# 每次執行時清除舊資料，確保新的 Metadata 生效
if os.path.exists("./chroma_langchain_db"):
    import shutil
    shutil.rmtree("./chroma_langchain_db")

vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding,
    persist_directory="./chroma_langchain_db"
)

# 4. 建立檢索器
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 15
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

# 5. 結構化重排序
@chain
def advanced_retriever(query: str) -> List[Document]:
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    # 去重
    unique_docs = {doc.page_content: doc for doc in bm25_docs + vector_docs}.values()
    
    scored_results = []
    for doc in unique_docs:
        score = 0
        meta = doc.metadata
        
        # 邏輯 A: 標題層級加分 (Level 2/3 是大章節，通常更重要)
        level = meta.get("level", 5)
        score += (6 - level) * 1.5 
        
        # 邏輯 B: Path 包含關鍵字 (這是最強的信號)
        path = str(meta.get("path", "")).lower()
        # 針對任何遊戲問題的「流程」或「準備」
        if any(word in path for word in ["準備", "流程", "規則", "setup", "說明書"]):
            if any(q in query for q in ["開始", "多少", "怎麼", "如何"]):
                score += 8.0 

        scored_results.append((score, doc))
    
    # 排序
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [doc for s, doc in scored_results[:10]]

# 6. 測試執行
print("================ 執行結構化加權檢索 (修正 IndexError) ================")
query = "遊戲一開始，每人要有多少張手牌"
results = advanced_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"[{i}] [Level: {doc.metadata.get('level')}] [Path: {doc.metadata.get('path')}]")
    print(f"{doc.page_content[:250]}...\n")
    print("-" * 50)