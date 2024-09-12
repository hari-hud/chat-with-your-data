# RAG based Question Answering System

RAG based Question Answering System to Chat With Your Data

## Setup

1. [Install Ollama](https://ollama.com/download)

2. Load model

```sh
ollama pull llama3.1:8b # general purpose model
ollama pull nomic-embed-text # text embedding model 
```

3. Install Python dependency
```
pip install -r requirement.txt
```

4. Run Server
```shell
streamlit run app.py
```


## References:
-  https://python.langchain.com/v0.2/docs/tutorials/local_rag/
