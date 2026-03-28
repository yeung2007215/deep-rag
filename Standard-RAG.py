from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()


#Get docuemnt from markdown file and load into ChromaDB
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# Embedding model config
embedding=OllamaEmbeddings(model="nomic-embed-text")


# Find ChromaDB collection, if not exist, create one
vector_store = Chroma(
    collection_name="asteriated_grail_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

# Tool for agent to retrieve information from ChromaDB
@tool
def get_rule(query:str) -> str:
    """Retrieve information"""
    retrieved_docs = vector_store.similarity_search(query, k=10)
    results = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return results

# Agent invoke
# question= input("請輸入你的問題: ")
question="星杯傳說, 起始手牌拿幾多張?"
similarity_search_results = get_rule.invoke({"query": question})


# Print("Test if retrieval success======")
if not similarity_search_results:
    print("No relevant information found in the database.")
else:
    print("================similarity_search_results================\n")
    print("retrieval success: Tool get_rule result:")
    print(similarity_search_results)


# Print AI reference
print("================AI references================\n")
retrieved_docs = vector_store.similarity_search(question, k=8)
for index, reference in enumerate(retrieved_docs):
    print(f"Index of reference:{index}")
    print(reference)
    print(f"================End {index}================\n")


print("================RAG Answers================")
# Agent system prompt
system_prompt = f"""
    你是桌遊規則助理。以下是從規則資料庫檢索到的內容：
    {similarity_search_results}

    請只根據以上內容回答：
    問題：{question}
"""

# Create agent
agent = create_agent(
    model="deepseek:deepseek-chat",
    # temperature=0.5,
    system_prompt=system_prompt,
    tools=[get_rule]
)

#RAG Result
results = agent.invoke({"messages":[{"role":"user", "content":question}]})
messages = results["messages"]
print("\nAI answer:", messages[-1].content)






# # # Loop to keep asking user for question
# while True:
#    user_input = input("\nYour question: ")
#    results=agent.invoke({"messages": [{"role": "user", "content": user_input}]})
   
#    messages = results["messages"]
#    print("\nAI:", messages[-1].content)

#    print("\n===========References============\n")
#    for result in messages:
#         print("\nReference(1):\n")
#         print(messages[0].content)
#         print("\n==============END==============\n")