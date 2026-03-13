import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 2})

def get_answer(query):
    try:
        retriever = load_retriever()
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)[:500]

        llm = ChatGroq(model="llama3-8b-8192", api_key=st.secrets["GROQ_API_KEY"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are GambiaGPT. Answer briefly using the context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})
    except Exception as e:
        return f"DEBUG ERROR: {str(e)}"
    llm = ChatGroq(model="mixtral-8x7b-32768", api_key=st.secrets["GROQ_API_KEY"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are GambiaGPT, an AI assistant for The Gambia. Use the context to answer concisely. If you don't know, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})

st.set_page_config(page_title="GambiaGPT", page_icon="🇬🇲")
st.title("🇬🇲 GambiaGPT")
st.write("Ask anything about The Gambia!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(query)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})