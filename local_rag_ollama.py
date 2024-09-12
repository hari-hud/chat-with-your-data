from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# load and split an example document
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
loader = PyPDFLoader(file_path = "hari-hud.pdf")
# loader = TextLoader("document.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# initialize vector store
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# set up a model
model = ChatOllama(
    model="llama3.1:8b",
)

def format_docs(docs):
    """
    Convert loaded documents into strings by concatenating their content and ignoring metadata
    :param docs:
    :return:
    """
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

# question = "What are the approaches to Task Decomposition?"
question = "Where does Hari Hud work?"

print(qa_chain.invoke(question))