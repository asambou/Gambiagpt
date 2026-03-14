import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

LEGAL_DOCS_PATH = "data/documents/legal"
LEGAL_VECTOR_PATH = "vectorstore_legal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_legal_documents():
    documents = []
    if not os.path.exists(LEGAL_DOCS_PATH):
        print(f"No legal docs found at {LEGAL_DOCS_PATH}")
        return documents
    for file in os.listdir(LEGAL_DOCS_PATH):
        filepath = os.path.join(LEGAL_DOCS_PATH, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif file.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue
            documents.extend(loader.load())
            print(f"  Loaded: {file}")
        except Exception as e:
            print(f"  Failed: {file} — {e}")
    return documents

def build_legal_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"\nTotal legal chunks: {len(chunks)}")

    print("Building legal FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(LEGAL_VECTOR_PATH, exist_ok=True)
    db.save_local(LEGAL_VECTOR_PATH)
    print(f"Legal vector store saved to: {LEGAL_VECTOR_PATH}")
    return db

if __name__ == "__main__":
    print("=== Loading Legal Documents ===")
    docs = load_legal_documents()
    print(f"Total legal pages loaded: {len(docs)}")

    if docs:
        print("\n=== Building Legal Vector Store ===")
        build_legal_vectorstore(docs)
        print("\nLegal ingestion complete!")
    else:
        print("No documents found. Run legal_downloader.py first.")