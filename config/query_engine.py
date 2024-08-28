def setup_query_engine(index, llm):
    return index.as_query_engine(
        llm=llm, 
        vector_store_query_mode="mmr", 
        vector_store_kwargs={"mmr_threshold": 0.2},
        streaming=True
    )
