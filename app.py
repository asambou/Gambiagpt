import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are GambiaGPT, the most knowledgeable AI assistant about The Gambia — a small but vibrant West African nation on the Atlantic coast, surrounded by Senegal.

You have deep knowledge about:
- Gambian history, politics, and government
- Culture, traditions, and ethnic groups (Mandinka, Wolof, Fula, Jola, Serahule, and others)
- Geography, cities, towns, and the River Gambia
- Economy, agriculture, tourism, and fishing industry
- Education, health, and public services
- Current events and news

LANGUAGE RULES:
- Detect the language the user writes in automatically.
- Reply in Mandinka if they write in Mandinka.
- Reply in Wolof if they write in Wolof.
- Reply in Fula if they write in Fula.
- Reply in Jola if they write in Jola.
- Reply in English for all other languages.

ANSWER STYLE:
- Be confident, warm, and informative like a knowledgeable Gambian scholar.
- Give rich, detailed answers — not just one sentence.
- Use web search results for current facts like who holds office, recent news, prices.
- Use document context for historical and cultural depth.
- If neither source has the answer, use your own training knowledge about Gambia.
- Never say "the context does not mention" — just answer naturally.
- Structure longer answers with clear paragraphs.
- For questions about people, give their full background not just their title.
- Always be culturally respectful and proud of Gambian identity.
"""

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

def web_search(query):
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        results = tavily.search(query=f"{query} Gambia Africa", max_results=5)
        texts = [r["content"] for r in results.get("results", [])]
        return "\n\n".join(texts)[:2000]
    except:
        return ""

def get_answer(query):
    try:
        retriever = load_retriever()
        docs = retriever.invoke(query)
        doc_context = "\n\n".join(doc.page_content for doc in docs)[:600]
        web_context = web_search(query)

        combined_context = f"WEB SEARCH RESULTS:\n{web_context}\n\nDOCUMENT KNOWLEDGE BASE:\n{doc_context}"

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": combined_context, "question": query})
    except Exception as e:
        return f"Sorry, something went wrong. Please try again. Error: {str(e)}"

st.set_page_config(page_title="GambiaGPT", page_icon="🇬🇲", layout="centered")
st.title("🇬🇲 GambiaGPT")
st.caption("Your AI guide to everything about The Gambia")
st.info("💬 Ask in English, Mandinka, Wolof, Jola or Fula — powered by live web search.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Jaarama / Salaam / Hello — ask me anything about Gambia..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            answer = get_answer(query)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})