# Building an Advanced Document QA System with LangChain and OpenAI

Tags: Python, LangChain, OpenAI, Vector Database, Natural Language Processing, Question Answering, Embeddings

Slug: advanced-document-qa-system-langchain-openai

Description: Learn how to create a powerful document question-answering system using Python, LangChain, and OpenAI's language models for efficient information retrieval and analysis.

---

In today's data-driven world, efficiently extracting relevant information from large documents is crucial. This blog post will guide you through building an advanced document question-answering (QA) system using Python, LangChain, and OpenAI's powerful language models. Our system will load text documents, create vector embeddings, store them in a vector database, and provide accurate answers to user queries.

## Key Components

1. Document Loading and Splitting
2. Vector Embeddings
3. Chroma Vector Database
4. Custom Retriever
5. Question-Answering Chain

## Project Structure

Our project consists of the following files:

- `main.py`: The main script that orchestrates the entire process
- `redundant_filter_retriever.py`: Implements a custom retriever for fetching relevant documents
- `prompt.py`: Contains utility functions for creating embeddings, managing the vector store, and setting up the question-answering system
- `CustomRetriever.py`: (Not provided in the code snippets, but mentioned in the file list)
- `Pipfile`: Defines the project dependencies
- `.env`: Contains environment variables (e.g., OpenAI API key)
- `facts.txt`: The input document containing the information to be queried

## Step 1: Setting Up the Environment

First, let's set up our project environment. Create a new directory for your project and install the required dependencies using Pipenv:

```bash
mkdir document_qa_system
cd document_qa_system
pipenv install langchain langchain-community langchain-openai langchain-chroma openai python-dotenv tiktoken chromadb
```

Create a `.env` file in your project directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Step 2: Loading and Splitting Documents

In `main.py`, we implement the document loading and splitting functionality:

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def load_and_split_document(file_path, chunk_size, chunk_overlap, separator):
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
    )
    return text_splitter.split_documents(docs)
```

This function uses `TextLoader` to load the document and `CharacterTextSplitter` to split it into manageable chunks.

### In-Depth: Document Loading and Splitting

The document loading and splitting process is crucial for efficient processing of large texts. Here's a closer look at the components:

- `TextLoader`: This LangChain utility loads the content of a text file. It's simple but effective for handling plain text documents.

- `CharacterTextSplitter`: This component breaks the document into smaller chunks. It's important because:
  1. It allows for more efficient processing of large documents.
  2. It enables more precise retrieval of relevant information.
  3. It helps in staying within token limits of language models.

The `chunk_size` parameter determines the maximum number of characters in each chunk, while `chunk_overlap` allows for some overlap between chunks to maintain context across splits. The `separator` parameter defines where to split the text, ensuring that splits occur at natural breakpoints in the document.

## Step 3: Creating Vector Embeddings and Storing in Chroma

In `prompt.py`, we define functions for creating embeddings and storing them in a Chroma vector database:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_embeddings_model():
    return OpenAIEmbeddings()

def store_in_chroma(chunks, embedding_model, persist_directory="db"):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
```

These functions create an OpenAI embeddings model and store the document chunks in a Chroma vector database.

### In-Depth: Vector Embeddings and Chroma Database

Vector embeddings and efficient storage are key to the performance of our QA system:

- **Vector Embeddings**: We use OpenAI's embedding model (`OpenAIEmbeddings`) to create vector representations of our text chunks. These embeddings capture the semantic meaning of the text, allowing for more accurate similarity searches. The embedding process converts text into high-dimensional vectors, where similar texts are closer together in the vector space.

- **Chroma Vector Database**: Chroma is a vector database that allows for efficient storage and retrieval of our document embeddings. It provides methods for similarity search, including the max marginal relevance search used in our custom retriever. Key features of Chroma include:
  1. Efficient indexing for fast similarity search
  2. Persistence of embeddings, allowing for reuse without recomputing
  3. Integration with LangChain for seamless use in our pipeline

The `store_in_chroma` function takes our document chunks, uses the embedding model to convert them to vectors, and stores these vectors in the Chroma database. The `persist_directory` parameter allows us to save the database to disk for future use.

## Step 4: Implementing a Custom Retriever

In `redundant_filter_retriever.py`, we implement a custom retriever to improve the relevance of retrieved documents:

```python
from pydantic import BaseModel
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class RedundantFilterRetriever(BaseRetriever, BaseModel):
    embeddings: Embeddings
    vector_store: Chroma

    def get_relevant_documents(self, query: str, num_results: int = 10) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.vector_store.max_marginal_relevance_search_by_vector(
            query_embedding,
            lambda_mult=0.5,
            k=num_results
        )
        return results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")
```

This custom retriever uses max marginal relevance search to reduce redundancy in retrieved documents.

### In-Depth: Custom Retriever

The `RedundantFilterRetriever` class is a crucial component that enhances the relevance and diversity of retrieved documents:

- **Inheritance**: By inheriting from `BaseRetriever` and `BaseModel`, it integrates seamlessly with LangChain's ecosystem and Pydantic's data validation.

- **Max Marginal Relevance (MMR) Search**: The `get_relevant_documents` method uses Chroma's `max_marginal_relevance_search_by_vector` function. MMR aims to maximize the relevance of the retrieved documents while minimizing redundancy. It does this by:
  1. Finding documents that are similar to the query
  2. Ensuring diversity among the selected documents

- **Customization**: The `lambda_mult` parameter (set to 0.5 here) balances between relevance and diversity. A higher value favors relevance, while a lower value promotes diversity.

- **Efficiency**: By using vector similarity search, this retriever can quickly find relevant documents even in large collections.

This custom retriever helps to provide a more comprehensive and less repetitive context for our question-answering system, potentially leading to more accurate and informative answers.

## Step 5: Setting Up the Question-Answering Chain

In `prompt.py`, we set up the question-answering chain using LangChain:

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def create_prompt():
    template = """
    You are a helpful assistant that can answer questions about the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate.from_template(template)

def create_retrieval_qa(retriever):
    prompt = create_prompt()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
```

These functions create a custom prompt and set up the retrieval QA system using our custom retriever.

### In-Depth: Question-Answering Chain

The question-answering chain is the core of our system, combining retrieval and language understanding:

- **Custom Prompt**: The `create_prompt` function defines a template that structures the input for the language model. This prompt engineering is crucial for guiding the model to generate relevant and accurate answers.

- **RetrievalQA Chain**: The `create_retrieval_qa` function sets up a `RetrievalQA` chain from LangChain. This chain:
  1. Uses the custom retriever to fetch relevant documents
  2. Incorporates these documents as context in the prompt
  3. Utilizes the ChatGPT model (`gpt-3.5-turbo`) to generate an answer

- **Chain Type**: The `"stuff"` chain type is used, which simply stuffs all retrieved documents into the prompt. This works well for situations where the relevant information can fit within the model's context window.

- **Source Documents**: By setting `return_source_documents=True`, we can trace which documents were used to generate the answer, enhancing explainability.

This setup allows for a powerful and flexible question-answering system that can adapt to various types of queries and documents.

## Step 6: Putting It All Together

In `main.py`, we create the main function to tie everything together:

```python
from dotenv import load_dotenv
from redundant_filter_retriever import RedundantFilterRetriever
from prompt import (
    store_in_chroma,
    query_loop,
    load_environment,
    create_embeddings_model,
    load_chroma_db_for_retrieval,
    create_retrieval_qa,
)

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
```

This main function loads the environment variables, processes the document, sets up the retrieval QA system, and runs an interactive query loop.

## Running the System

To run the system, make sure you have a `facts.txt` file in your project directory containing the text you want to query. Then, execute the following command:

```bash
python main.py
```

You can now enter questions about the content in `facts.txt`, and the system will provide answers based on the document's content.

## Conclusion

We've built an advanced document QA system using Python, LangChain, and OpenAI's language models. This system efficiently processes documents, creates vector embeddings, and uses a custom retriever to provide accurate answers to user queries.

The combination of LangChain's abstractions, OpenAI's powerful language models, and Chroma's efficient vector storage allows for the creation of a system that can understand and answer questions about large documents with high accuracy and relevance.

This project demonstrates the power of combining various NLP techniques and tools to create a sophisticated information retrieval and analysis system. It can be adapted for various use cases, from personal knowledge management to enterprise-level document analysis.

Future improvements could include multi-document support, improved error handling, a web interface for easier interaction, or integration with other data sources. The modular nature of this system allows for easy expansion and customization to meet specific needs.

## Bibliography and Resources

1. LangChain Documentation. (n.d.). Retrieved from <https://python.langchain.com/docs/get_started/introduction>

2. OpenAI API Documentation. (n.d.). Retrieved from <https://platform.openai.com/docs/introduction>

3. Chroma Documentation. (n.d.). Retrieved from <https://docs.trychroma.com/>

4. Pydantic Documentation. (n.d.). Retrieved from <https://docs.pydantic.dev/latest/>

5. Pipenv Documentation. (n.d.). Retrieved from <https://pipenv.pypa.io/en/latest/>

6. Python-dotenv Documentation. (n.d.). Retrieved from <https://github.com/theskumar/python-dotenv>

7. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv preprint arXiv:1908.10084.

8. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval (pp. 335-336).

9. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

10. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.

These resources provide additional information on the key technologies and concepts used in this project, including LangChain, OpenAI's language models, vector databases, and relevant research papers on embeddings and retrieval methods.
