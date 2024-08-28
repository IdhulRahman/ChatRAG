import os
import time
from config.llm_conf import create_llm
from config.doc_Loader import load_documents
from config.embed_model import create_embedding_model, create_and_save_optimum_model
from config.vector_index import create_index
from config.query_engine import setup_query_engine
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

def chat_loop(query_engine):
    print("Start chatting with the system! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break
        
        start_time = time.time() 

        response = query_engine.query(user_input)
        response.print_response_stream()

        end_time = time.time() 
        response_time = end_time - start_time  

        print(f"Response time: {response_time:.2f} seconds")

def main():
    # Check if the embedding model folder exists
    embedding_folder = "mxbai-rerank"
    if os.path.exists(embedding_folder):
        embed_model = create_embedding_model()
    else:
        create_and_save_optimum_model()
        embed_model = create_embedding_model()
    
    Settings.embed_model = embed_model

    # Configure the LLM
    llm = create_llm()
    Settings.llm = llm

    # Check if the index folder already exists
    index_folder = "datavector"
    if not os.path.exists(index_folder):
        # Load the documents
        documents = load_documents(["dataset/Data3.txt"])

        # Create the index from documents
        index = create_index(documents, embed_model)
    else:
        print(f"Folder '{index_folder}' exists, loading index from storage.")
        
        # Rebuild storage context and load index
        storage_context = StorageContext.from_defaults(persist_dir=index_folder)
        index = load_index_from_storage(storage_context, index_id="vector_index")
        
    # Set up the query engine
    query_engine = setup_query_engine(index, llm)

    # Start the chat loop
    chat_loop(query_engine)

if __name__ == "__main__":
    main()
