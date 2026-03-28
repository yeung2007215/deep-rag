from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
embedding=OllamaEmbeddings(model="nomic-embed-text")

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。 ", "！"],
    chunk_size=200,
    chunk_overlap=50,
    add_start_index=True
)


# ==========================================
# Collection 1: 星杯傳說 (Asteriated Grail)
# ==========================================
file_path_1 = "docs/星杯傳說_規則說明書.md"
loader_1 = UnstructuredMarkdownLoader(file_path_1)
docs_1 = loader_1.load()
all_splits_1 = text_splitter.split_documents(docs_1)

vector_store_1 = Chroma(
    collection_name="asteriated_grail_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

print(f"\nStoring Asteriated Grail(星杯傳說)...")
vector_store_1.add_documents(documents=all_splits_1)
print(f"Success!")
print(f"Collection Name: asteriated_grail_collection")
print(f"Chunks: {len(all_splits_1)}")


# ===============================================
# Collection 2: 三國殺 (War of the Three Kingdoms)
# ===============================================
file_path_2 = "docs/三國殺_規則說明書.md" 
try:
    loader_2 = UnstructuredMarkdownLoader(file_path_2)
    docs_2 = loader_2.load()
    all_splits_2 = text_splitter.split_documents(docs_2)

    vector_store_2 = Chroma(
        collection_name="war_of_the_three_kingdom_collection",
        embedding_function=embedding,
        persist_directory="./chroma_langchain_db"
    )

    print(f"\n--- Storing War of the Three Kingdoms(三國殺) ---")
    vector_store_2.add_documents(documents=all_splits_2)
    print(f"Success!")
    print(f"Collection Name: war_of_the_three_kingdom_collection")
    print(f"Chunks: {len(all_splits_2)}")
except Exception as e:
    print(f"\nFailed to process War of the Three Kingdoms document (please check if the file exists): {e}")


print("\n~~~~~~~~~~~~~~~~ Indexing Completed ~~~~~~~~~~~~~~~~\n")



# ids = vector_store.add_documents(documents=all_splits)
# print("Doc IDs added:")
# print(len(ids)) #18
# print(ids) #id



