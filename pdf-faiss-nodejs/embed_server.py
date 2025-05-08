from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

# CORS for Node.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Texts(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed_text(data: Texts):
    embeddings = model.encode(data.texts, convert_to_numpy=True).tolist()
    print("Generated embeddings:", embeddings)  # Add logging here to check the embeddings
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run("embed_server:app", host="0.0.0.0", port=3000, reload=True)
