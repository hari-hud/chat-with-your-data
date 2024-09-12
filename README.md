# RAG based Question Answering System

RAG based Question Answering System to Chat With Your Data

## Setup

1. Install Ollama: https://ollama.com/download

2. load model
```
ollama pull llama3.1:8b # general purpose model
ollama pull nomic-embed-text # text embedding model 
```

3. Install Python packages:
```
pip install langchain langchain_community  langchain_chroma langchain_ollama beautifulsoup4 streamlit
```

4. Run Server
```shell
streamlit run app.py
```


## References:
-  https://python.langchain.com/v0.2/docs/tutorials/local_rag/
