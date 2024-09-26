from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma

memory_key = "history"

import os

# Directories
chromadb = "db"
upload_folder = "upload"
os.makedirs(upload_folder, exist_ok=True)

client = Chroma(persist_directory=chromadb)
dblist = client.get()
embedded_docs = sources = [item['source'] for item in dblist['metadatas']]
print(embedded_docs)

# recursively traverse the upload folder and load the documents into the vector store
for root, dirs, files in os.walk(upload_folder):
    for file in files:
        file_path = os.path.join(root, file)
        
        if file_path.endswith('.md'):
            text_chunks = UnstructuredMarkdownLoader(file_path).load_and_split()
        else:
            continue
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=chromadb)
        print(f"Loaded {file_path} into vector store.")
