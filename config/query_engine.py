from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

def setup_query_engine(index, llm):
    rerank = FlagEmbeddingReranker(model="mixedbread-ai/mxbai-rerank-large-v1", top_n=3)
    
    return index.as_query_engine(
        llm=llm,
        node_postprocessors=[rerank], 
        vector_store_query_mode="mmr", 
        vector_store_kwargs={"mmr_threshold": 0.2},
        streaming=True
    )
