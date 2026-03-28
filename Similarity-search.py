from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


embedding=OllamaEmbeddings(model="nomic-embed-text")

vector_store=Chroma(
    collection_name="asteriated_grail_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=10)


print("================用檢索器進行相似度查詢================")
results = retriever.invoke("遊戲一開始，每人要有多少張手牌")
print("Similarity Search Done")

for index, result in enumerate(results):  #unpacking
    print(f"[========Index of result:{index+1}========]\n")
    print(result.page_content[:500])
    print(f"[End of chunk no.{index+1}]\n")





# query = "星杯傳說, 起始手牌拿幾張?"
# results_with_score = vector_store.similarity_search_with_score(query, k=3)

# for doc, score in results_with_score:
#     print(f"【相似度分數】: {score:.4f}") # 數值越小越好
#     print(f"內容: {doc.page_content[:100]}...")
#     print("-" * 30)
