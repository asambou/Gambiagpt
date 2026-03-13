from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTOR_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """Use the following context to answer the question.
If you don't know, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
llm = Ollama(model="llama3")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("\nAsk a question (or type 'quit' to exit): ")
    if query.lower() == "quit":
        break
    answer = chain.invoke(query)
    print("\nAnswer:", answer)