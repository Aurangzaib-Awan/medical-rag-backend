from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

def load_db():
    emb=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db=FAISS.load_local(
        "my_langchain_index",
        emb,
        allow_dangerous_deserialization=True
    )
    return db