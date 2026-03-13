import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
DATA_PATH = "data/documents"
VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(data_path: str):
    """Load PDF and TXT files from a directory."""
    documents = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    for file in os.listdir(data_path):
        filepath = os.path.join(data_path, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif file.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue  # Skip unsupported file types
            documents.extend(loader.load())
            print(f"  ✔ Loaded: {file}")
        except Exception as e:
            print(f"  ✘ Failed to load {file}: {e}")

    return documents

def split_documents(documents):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def build_vectorstore(docs, vector_path: str):
    """Create and save a FAISS vector store from document chunks."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Building FAISS index...")
    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(vector_path, exist_ok=True)
    db.save_local(vector_path)
    print(f"✔ Vector store saved to: {vector_path}")
    return db

if __name__ == "__main__":
    print("=== Step 1: Loading Documents ===")
    documents = load_documents(DATA_PATH)
    print(f"Total pages/docs loaded: {len(documents)}")

    print("\n=== Step 2: Splitting into Chunks ===")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("\n=== Step 3: Creating Vector Store ===")
    build_vectorstore(chunks, VECTOR_PATH)
    print("\n✅ Ingestion complete!")