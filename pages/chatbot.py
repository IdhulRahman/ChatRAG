import os
import streamlit as st
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

# Function to initialize the chatbot engine
def initialize_chatbot():
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
        
    query_engine = setup_query_engine(index, llm)
    return query_engine

# Function to handle chatbot response
def chat_with_bot(query_engine, user_input):
    response = query_engine.query(user_input)
    response_text = ""
    if hasattr(response, "response_stream"):
        for chunk in response.response_stream:
            response_text += chunk
    else:
        response_text = str(response)
    return response_text

# Streamlit page for chatbot with bubble chat interface
def chatbot_page():
    st.title("LLM Chatbot Interface")
    st.write("Start chatting with the system below!")

    # CSS for bubble chat style
    st.markdown("""
        <style>
        .user-bubble {
            background-color: #DCF8C6;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 70%;
            text-align: left;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #ECECEC;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 70%;
            text-align: left;
            float: left;
            clear: both;
        }
        .chat-container {
            display: flex;
            flex-direction: column; /* Keep normal order, new messages will appear below */
            margin-bottom: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize chatbot engine
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = initialize_chatbot()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history in the chat column
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:  # Messages displayed in normal order (new at the bottom)
            st.markdown(f'<div class="user-bubble">You: {msg["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-bubble">Bot: {msg["bot"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Use a temporary variable to capture user input
    user_input = st.text_input("Type your message here and press Enter:")

    # Automatically send message when user presses Enter or Send button
    if st.button("Send") or user_input:
        if user_input:
            response = chat_with_bot(st.session_state.query_engine, user_input)
            st.session_state.messages.append({"user": user_input, "bot": response})
            st.rerun()  # Re-run the app to update UI with new messages

# Main function to run the page
if __name__ == "__main__":
    chatbot_page()
