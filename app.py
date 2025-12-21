from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from google import genai
import os


load_dotenv()
GENAI_KEY = os.getenv("GENAI_KEY")


emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local(
    "my_langchain_index",
    emb,
    allow_dangerous_deserialization=True
)

client = genai.Client(api_key=GENAI_KEY)





if query:
    docs = db.similarity_search(query, k=3)
   
    # Display snippet for each doc (first 50–100 chars)
    st.write("### Retrieved Context Snippets:")

    num = 1
    for doc in docs:
        text = doc.page_content.strip()
        text = text.replace("\n", " ")

        if len(text) > 100:
            text = text[:100] + "..."

        st.write(num, text)
        num += 1

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Context:
    {context}

    Question: {query}
    You're a helpfull medical assistant, answer the queries by analysing the given context,it is not necessary that what context says is applied to pateint but could be a possibility,
    keep the replies short if context doesnt specify anything simply say not enough context to answer this:
    """

    resp = client.models.generate_content(
        model="models/gemma-3-4b-it",
        contents=prompt
    )

    st.write("### Answer:")
    st.write(resp.text)
