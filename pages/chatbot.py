import os
from fastapi import FastAPI
from pydantic import BaseModel
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

# Initialize FastAPI app
app = FastAPI()

# Define request body model
class ChatRequest(BaseModel):
    user_input: str

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

# Initialize chatbot engine when app starts
query_engine = initialize_chatbot()

# Function to handle chatbot response
def chat_with_bot(user_input):
    response = query_engine.query(user_input)
    response_text = ""
    if hasattr(response, "response_stream"):
        for chunk in response.response_stream:
            response_text += chunk
    else:
        response_text = str(response)
    return response_text

# POST endpoint to interact with the chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    response = chat_with_bot(user_input)
    return {"user_input": user_input, "response": response}

# GET endpoint for testing server is live
@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}
