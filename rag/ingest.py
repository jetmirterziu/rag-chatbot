import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DATA_PATH = "data"
DB_PATH = "vector_db"

def ingest_docs():
    # 1. Clear old database
    if os.path.exists(DB_PATH):
        print(f"Removing old database at '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)

    # 2. Load PDFs
    print(f"Loading PDFs from '{DATA_PATH}'...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"   -> Loaded {len(raw_documents)} raw pages.")

    # 3. Split (Keeping 'add_start_index=True' for future highlighting)
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   -> Created {len(chunks)} chunks.")

    # 4. Embed & Save
    print("Creating vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if i == 0:
            vectorstore = Chroma.from_documents(batch, embedding_model, persist_directory=DB_PATH)
        else:
            vectorstore.add_documents(batch)
            
    print(f"Success! Vector store saved to '{DB_PATH}'.")

if __name__ == "__main__":
    ingest_docs()