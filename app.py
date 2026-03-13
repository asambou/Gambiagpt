import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """Use the context below to answer the question in 2-3 sentences.
If you don't know, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    llm = ChatGroq(model="llama3-8b-8192", api_key=st.secrets["GROQ_API_KEY"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

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
            chain = load_chain()
            answer = chain.invoke(query)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})