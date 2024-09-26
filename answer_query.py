import warnings
warnings.filterwarnings('ignore')

from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

local_llm = "./llama/llama-2-7b-chat.Q5_K_M.gguf"
chromadb = "db"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=chromadb, embedding_function=embeddings)

llm = LlamaCpp(
        model_path=local_llm,
        n_batch=1024,
        n_ctx=4096,
        f16_kv=True,  # MUST set to True, otherwise problem after a couple of calls
        # callback_manager=callback_manager,
        verbose=True,
    )
qa = RetrievalQA.from_llm(llm, retriever=vectorstore.as_retriever())

# Until quit using the command 'q', the user can input a query in terminal 
print("Enter 'q' to quit.")
while True:
    query = input("Enter a query: ")
    if query.lower() == 'q':
        break
    else:
        results = qa(query)
        answer = results['result']
        print("Answer: ", answer)


llm.close()