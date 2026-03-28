from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker

#load file - 規則說明書
file_path = "docs/星杯傳說_規則說明書.md"
loader = UnstructuredMarkdownLoader(file_path)
docs = loader.load()

#Chunk document
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
separators=[ "\n\n", "\n", "。 ", "！", "？", " ", ""],
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)


all_splits=text_splitter.split_documents(docs)

#Define Embedding model
from langchain_ollama import OllamaEmbeddings
embedding=OllamaEmbeddings(model="nomic-embed-text")

#Create ChromaDB, indexing
vector_store=Chroma(
    collection_name="asteriated_grail_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

print("~~~~~~~~~~~~~~~~Indexing Document into ChromaDB~~~~~~~~~~~~~~~~\n")
print("Save Result:")
print("Data saved in to ChromaDB, Collection name: starcup_collection")
print("Number of chunks:")
print(len(all_splits)) #159 chunks

ids = vector_store.add_documents(documents=all_splits)
# print("Doc IDs added:")
# print(len(ids)) #18
# print(ids) #id



