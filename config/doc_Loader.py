from llama_index.core import SimpleDirectoryReader

def load_documents(file_paths: list[str]):
    reader = SimpleDirectoryReader(input_files=file_paths)
    return reader.load_data()
