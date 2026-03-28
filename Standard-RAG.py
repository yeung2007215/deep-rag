from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

#Get docuemnt from markdown file and load into ChromaDB
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# Embedding model config
embedding=OllamaEmbeddings(model="nomic-embed-text")
current_vector_store = None

# Tool for agent to retrieve information from ChromaDB
@tool
def get_rule(query: str) -> str:
    """從選定的桌遊規則資料庫中檢索資訊。"""
    if current_vector_store is None:
        return "尚未選擇遊戲，無法檢索。"
    
    # Get the top 10 chunks most relevant to the query 
    retrieved_docs = current_vector_store.similarity_search(query, k=10)
    results = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return results


print("=== 🦘  Welcome to use BoardgameRoo 🦘 ===")

# 1st loop：select collection
while True:
    print("\n🕹️ Game list:")
    print("[1] ⭐️ 星杯傳說 (asteriated_grail_collection)")
    print("[2] ⚔️  三國殺 (war_of_the_three_kingdom_collection)")
    print("Input 'exit' to exit the program")
    
    choice = input("\n🦘: Enter game number or name (1/2): ").strip()

    if choice.lower() in ['exit', 'quit', '退出']:
        break

    # Selecting collection by user input
    if choice == "1" or "星杯" in choice:
        target_collection = "asteriated_grail_collection"
        game_name = "星杯傳說"
    elif choice == "2" or "三國殺" in choice:
        target_collection = "war_of_the_three_kingdom_collection"
        game_name = "三國殺"
    else:
        print("[Error] Invalid choice, please retry.")
        continue

    # Init Vector Store
    current_vector_store = Chroma(
        collection_name=target_collection,
        embedding_function=embedding,
        persist_directory="./chroma_langchain_db"
    )
    
    print(f"\n🦘 You are in[{game_name}]")
    print(f"(Enter'back' to return to game selection, 'exit' to quit program)")

    # 2nd loop：focus on the selected collection, ask questions
    while True:
        question = input(f"\n[{game_name}] Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() == 'back':
            print("Back to game list...")
            break
        
        if question.lower() in ['exit', 'quit', '退出']:
            print("Programme exiting...")
            exit()

        # Execute similarity search
        similarity_search_results = get_rule.invoke({"query": question})

        if not similarity_search_results.strip():
            print("警告：在資料庫中找不到相關片段，AI 將嘗試以現有知識回答。")

        # Set system prompt
        system_prompt = f"""
            你是《{game_name}》的桌遊規則助理。
            以下是從該遊戲規則資料庫檢索到的相關內容：
            {similarity_search_results}

            指令：
            1. 請優先根據以上提供的檢索內容回答問題。
            2. 如果檢索內容中完全沒有提到相關資訊，請老實回答「在目前的規則書中找不到相關說明」。
            3. 回答必須準確且語氣專業。
        """

        # Create Agent
        agent = create_agent(
            model="deepseek:deepseek-chat",
            system_prompt=system_prompt,
            tools=[get_rule]
        )
        
        try:
            print("🤖I am thinking...")
            results = agent.invoke({"messages": [{"role": "user", "content": question}]})
            
            ai_answer = results["messages"][-1].content
            
            print("\n========= AI Answer =========")
            print(ai_answer)
            print("========= End =========\n")

            
            
            # ==============================================================
            # Only for develpr testing: print retrieved chunks for reference
            retrieved_docs = current_vector_store.similarity_search(question, k=10)
            print(f"\n[AI reference]: Seacrhed chunks: {len(retrieved_docs)} ")
            for index, doc in enumerate(retrieved_docs):
                print(f"=====[AI reference: Chunk no.{index+1}]=====")
                # Print first 500 characters, and add source information (if available)
                source = doc.metadata.get('source', 'Unknown source')
                print(f"Source: {source}")
                print(f"Content: {doc.page_content[:500]}...") 
                print(f"[End of chunk no.{index+1}]\n")
            

        except Exception as e:
            print(f"Error occurred: {e}")