from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from redundant_filter_retriever import RedundantFilterRetriever


TEMPLATE = """
You are a helpful assistant that can answer questions about the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

def create_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(TEMPLATE)

def create_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()

def load_environment() -> None:
    load_dotenv()

def store_in_chroma(chunks: List[Document], embedding_model: OpenAIEmbeddings, persist_directory: str = "db") -> VectorStore:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

def load_chroma_db_for_retrieval(embedding_model: OpenAIEmbeddings, persist_directory: str = "db") -> BaseRetriever:
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return RedundantFilterRetriever(
        embeddings=embedding_model,
        vector_store=vector_store
    )

def search_in_chroma(retriever: BaseRetriever, query: str, k: int = 1) -> List[Document]:
    return retriever.get_relevant_documents(query, k=k)

def create_retrieval_qa(retriever: BaseRetriever) -> RetrievalQA:
    prompt = create_prompt()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

def query_loop(retrieval_qa: RetrievalQA, show_sources: bool = False) -> None:
    while True:
        query = input("Enter a query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        result = retrieval_qa.invoke({"query": query})
        print("Answer:", result['result'])
        if show_sources:
            print("\nSources:")
            for doc in result['source_documents']:
                print(f"- {doc.page_content[:100]}...")
