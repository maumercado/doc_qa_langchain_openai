from typing import Tuple
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from redundant_filter_retriever import RedundantFilterRetriever
from prompt import (
    store_in_chroma,
    query_loop,
    load_environment,
    create_embeddings_model,
    load_chroma_db_for_retrieval,
    create_retrieval_qa,
)

def load_and_split_document(file_path: str, chunk_size: int, chunk_overlap: int, separator: str):
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
    )
    return text_splitter.split_documents(docs)

def initialize_or_load_chroma(file_path: str, chunk_size: int, chunk_overlap: int, separator: str, persist_directory: str) -> Tuple[VectorStore, OpenAIEmbeddings]:
    embedding_model = create_embeddings_model()

    # Try to load existing Chroma DB
    try:
        vector_store = load_chroma_db_for_retrieval(embedding_model, persist_directory)
        print("Loaded existing Chroma DB.")
    except Exception as e:
        print(f"Error loading existing Chroma DB: {e}")
        print("Creating new Chroma DB and loading documents...")
        chunks = load_and_split_document(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator
        )
        vector_store = store_in_chroma(chunks, embedding_model, persist_directory)
        print("New Chroma DB created and documents loaded.")

    return vector_store, embedding_model

def main():
    load_environment()
    embedding_model = create_embeddings_model()

    try:
        retriever = load_chroma_db_for_retrieval(embedding_model)
        print("Existing Chroma DB loaded successfully.")
    except Exception as e:
        print(f"Error loading existing Chroma DB: {e}")
        print("Creating new Chroma DB and loading documents...")
        chunks = load_and_split_document(
            "./facts.txt",
            chunk_size=150,
            chunk_overlap=0,
            separator="\n"
        )
        vector_store = store_in_chroma(chunks, embedding_model, persist_directory="db")
        retriever = RedundantFilterRetriever(embeddings=embedding_model, vector_store=vector_store)
        print("New Chroma DB created and documents loaded.")

    retrieval_qa = create_retrieval_qa(retriever)
    query_loop(retrieval_qa, show_sources=True)

if __name__ == "__main__":
    main()
