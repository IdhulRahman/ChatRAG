from llama_index.core import VectorStoreIndex

def create_index(documents, embed_model) -> VectorStoreIndex:
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)
    index.set_index_id("vector_index")
    index.storage_context.persist("datavector")
    return index
