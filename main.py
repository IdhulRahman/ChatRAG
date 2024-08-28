import streamlit as st
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

# Initialize models and index
def initialize_models_and_index():
    embedding_folder = "mxbai-rerank"
    if os.path.exists(embedding_folder):
        embed_model = create_embedding_model()
    else:
        create_and_save_optimum_model()
        embed_model = create_embedding_model()
    
    Settings.embed_model = embed_model

    llm = create_llm()
    Settings.llm = llm

    index_folder = "datavector"
    if not os.path.exists(index_folder):
        documents = load_documents(["dataset/Data3.txt"])
        index = create_index(documents, embed_model)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=index_folder)
        index = load_index_from_storage(storage_context, index_id="vector_index")
    
    return setup_query_engine(index, llm)

def main():
    st.title("Chatbot Interface")

    # Initialize query engine
    query_engine = initialize_models_and_index()

    st.write("Start chatting with the system! Type 'exit' to end the chat.")

    user_input = st.text_input("You:", "")

    if st.button("Submit"):
        if user_input.lower() == "exit":
            st.write("Exiting chat. Goodbye!")
        else:
            with st.spinner('Regenerating response...'):
                start_time = time.time()
                response = query_engine.query(user_input)
                response_text = response.get_text()  # Assuming this method gets the text
                response_time = time.time() - start_time
                
                st.write("Response:")
                st.write(response_text)
                st.write(f"Response time: {response_time:.2f} seconds")

if __name__ == "__main__":
    main()
