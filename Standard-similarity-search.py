from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


embedding=OllamaEmbeddings(model="nomic-embed-text")

vector_store=Chroma(
    collection_name="starcup_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

print("================用檢索器進行相似度查詢================\n")
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=8)


results = retriever.invoke("遊戲一開始，每人要有多少張手牌")

for index, result in enumerate(results):  #unpacking
    print(f"Index of result:{index}")
    print(result.page_content[:200])
    print("================End================\n")







# results = vector_store.similarity_search(
#     "一開始遊戲設置，每人要有多少張手牌?"
# )


# for index, result in enumerate(results):
#     print(f"Index of result:{index}")
#     print("Print out page content 100 Char:")
#     print(result.page_content[:100])
#     # result 是 page content, metadata
#     #print(result)
#     #[:100] 只取100個字符
#     print("Similarity Search Done")
#     print("\n")
#     print("\n")
