from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
import streamlit as st

def load_file(file):
    if file.type == "application/pdf":
        print(file.name)
        return PyPDFLoader(file_path=file.name).load()
    elif file.type == "text/plain":
        return TextLoader(file_path=file.name).load()
    else:
        st.error("Unsupported file type")
        return []


# Streamlit UI
st.title("Local RAG Question Answering System")

file = st.file_uploader("Upload a PDF, Word, or Text file", type=["pdf", "txt"])

if file:
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())

    data = load_file(file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Initialize vector store
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    # Set up a model
    model = ChatOllama(model="llama3.1:8b")


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    retriever = vectorstore.as_retriever()

    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | model
            | StrOutputParser()
    )

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            answer = qa_chain.invoke(question)
            st.write("Answer:", answer)
        else:
            st.warning("Please enter a question.")