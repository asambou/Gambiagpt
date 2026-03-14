import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are GambiaGPT, an AI assistant for The Gambia.

LANGUAGE RULES:
- Detect the language the user writes in.
- If they write in Mandinka, respond in Mandinka.
- If they write in Wolof, respond in Wolof.
- If they write in Fula, respond in Fula.
- If they write in Jola, respond in Jola.
- Otherwise respond in English.

ANSWER RULES:
- Use the web search results first for current information.
- Use the document context to add extra detail.
- Keep answers concise and helpful.
- Be respectful and culturally sensitive to Gambian culture.
"""

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 2})

def web_search(query):
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        results = tavily.search(query=f"{query} Gambia", max_results=3)
        texts = [r["content"] for r in results.get("results", [])]
        return "\n\n".join(texts)[:1000]
    except:
        return ""

def get_answer(query):
    try:
        retriever = load_retriever()
        docs = retriever.invoke(query)
        doc_context = "\n\n".join(doc.page_content for doc in docs)[:400]
        web_context = web_search(query)

        combined_context = f"WEB SEARCH RESULTS:\n{web_context}\n\nDOCUMENT CONTEXT:\n{doc_context}"

        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": combined_context, "question": query})
    except Exception as e:
        return f"Sorry, something went wrong. Please try again."

st.set_page_config(page_title="GambiaGPT", page_icon="🇬🇲")
st.title("🇬🇲 GambiaGPT")
st.write("Ask anything about The Gambia — in English, Mandinka, Wolof, Jola or Fula!")
st.info("💬 Powered by live web search + Gambia document knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("Jaarama / Salaam / Hello / Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            answer = get_answer(query)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})