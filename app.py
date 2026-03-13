import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- Configuration ---
VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Initialize Groq LLM ---
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key=st.secrets["GROQ_API_KEY"]
)

# --- Load FAISS retriever ---
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 2})

retriever = load_retriever()

# --- Function to get AI answer ---
def get_answer(query: str) -> str:
    # Get relevant docs
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)[:3000]  # truncate to avoid Groq limits

    # Prepare the system + user prompt
    messages = [
        {"role": "system", "content": "You are GambiaGPT, an AI assistant for The Gambia. Use the context to answer. If you don't know, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    # Generate response
    result = llm.generate(messages=messages)
    
    # Extract text (depends on Groq SDK, usually in 'generations')
    answer_text = result.generations[0].text if hasattr(result, "generations") else str(result)
    return answer_text

# --- Streamlit UI ---
st.set_page_config(page_title="GambiaGPT", page_icon="🇬🇲")
st.title("🇬🇲 GambiaGPT")
st.write("Ask anything about The Gambia!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(query)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
