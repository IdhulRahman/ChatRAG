import os
import torch
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding

def create_embedding_model() -> OptimumEmbedding:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return OptimumEmbedding(folder_name="mxbai-rerank", device=device)

# mixedbread-ai/mxbai-rerank-large-v1
# mixedbread-ai/mxbai-rerank-base-v1

def create_and_save_optimum_model():
    OptimumEmbedding.create_and_save_optimum_model(
        "mixedbread-ai/mxbai-rerank-large-v1", "./mxbai-rerank", 
    )

def main():
    if os.path.exists("mxbai-rerank"):
        embedding_model = create_embedding_model()
    else:
        create_and_save_optimum_model()
        embedding_model = create_embedding_model()

    return embedding_model