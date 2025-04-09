import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# Function to load and split PDF text
async def load_and_split_pdf(file_path: str, chunk_size=250, chunk_overlap=50):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    
    complete_text = "".join(page.page_content for page in pages)
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    texts = text_splitter.create_documents([complete_text])
    return texts

# Function to build or load a vector store
def build_vector_store(docs, persist_directory="chroma_db", embedding_model="llama3.2"):
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = OllamaEmbeddings(model=embedding_model)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore

async def rag_qa_pipeline(file_path: str):
    print("📄 Loading and splitting PDF...")
    texts = await load_and_split_pdf(file_path)

    print("💾 Building and saving vector store...")
    vectorstore = build_vector_store(texts)

    return "Data Preparation completed"

import asyncio

if __name__ == "__main__":
    answer = asyncio.run(
        rag_qa_pipeline("RAG-Development/data/ChatGPT_Workbook.pdf")
    )

