from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient
import os

app = Flask(__name__)

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SYSTEM_PROMPT = """You are GambiaGPT, the most knowledgeable AI assistant about The Gambia.
You are running on WhatsApp so keep answers concise — maximum 3 paragraphs.
Answer in the same language the user writes in.
Cover: history, government, tourism, health, education, cybersecurity, networking.
Always be warm, helpful and proud of Gambian identity."""

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 2})

def web_search(query):
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        results = tavily.search(query=f"{query} Gambia", max_results=3)
        texts = [r["content"] for r in results.get("results", [])]
        return "\n\n".join(texts)[:1500]
    except:
        return ""

def get_answer(query):
    try:
        docs = retriever.invoke(query)
        doc_context = "\n\n".join(doc.page_content for doc in docs)[:400]
        web_context = web_search(query)
        combined_context = f"WEB:\n{web_context}\n\nDOCS:\n{doc_context}"

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": combined_context, "question": query})
    except Exception as e:
        return "Sorry, I could not answer that. Please try again."

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")

    print(f"Message from {sender}: {incoming_msg}")

    answer = get_answer(incoming_msg)

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(answer)

    return str(resp)

@app.route("/", methods=["GET"])
def health():
    return "GambiaGPT WhatsApp Bot is running!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)