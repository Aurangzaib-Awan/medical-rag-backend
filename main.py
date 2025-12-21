import os
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from google import genai
from fastapi.middleware.cors import CORSMiddleware
from vector import load_db

load_dotenv()
GENAI_KEY = os.getenv("GENAI_KEY")
if not GENAI_KEY:
    raise HTTPException("Api key not found")

#intializing the gemini client
client=genai.Client(api_key=GENAI_KEY)

#base model for prompt
class Query(BaseModel):
    prompt:str

app=FastAPI()

@app.get("/health")   
def health():
    return {"status": "ok"}

#did cors so react can be recognized
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the db from vector.py
db=load_db()


@app.post("/query")
def query_gemini(request:Query):
    """
    1. Get the user query from request
    2. Search for top-k similar documents from the LangChain DB
    3. Combine context and user query into a prompt
    4. Call Gemini API with the combined prompt
    5. Return the response
    """

    try:
        prompt=request.prompt
        docs=db.similarity_search(query=prompt,k=3)
        context="\n\n".join([d.page_content for d in docs])

        #now lets create snippets so we can show what we retrieve
        context_retrieved=[]
        for i,doc in enumerate(docs,start=1):
            text = doc.page_content.strip().replace("\n", " ")
            text = text[:100] + "..." if len(text)>100 else text
            context_retrieved.append(f"{i} . {text}")

        #now create final prompt 
        final_prompt = f"""
You are a helpful medical AI assistant.

The following REFERENCE CONTEXT is **general medical knowledge** retrieved from a knowledge base.
It may contain information about various medical cases, but it does NOT describe the user's personal condition.

Your job:
- Use the context ONLY as supporting knowledge
- Do NOT assume the user has any conditions mentioned in the context
- Do NOT give a medical diagnosis
- Do NOT give a disclaimer
- Do NOT asks for more follow-up, clarrifying questions or invite user to continue conversation   
- Provide safe, general, and empathetic advice for the user's question
- Highlight important warning signs or next steps in general terms
- Use bullet points for clarity

REFERENCE CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUESTION:
\"\"\"
{prompt}
\"\"\"

Answer clearly, concisely, and safely, giving general guidance but without diagnosing the user.
"""

        response=client.models.generate_content(
            model="models/gemma-3-4b-it",
            contents=final_prompt
            )
        return{
            "prompt_recieved":final_prompt,
            "context_retrieved":context_retrieved,
            "gemini_response":response.text,
            "disclaimer": "This information is for educational purposes only and is not a substitute for professional medical advice."

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        