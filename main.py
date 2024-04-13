from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import OllamaClient

from dotenv import load_dotenv

import os

load_dotenv()

ollama_client = None
context = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_client
    ollama_client = OllamaClient(os.getenv("OLLAMA_URL"), os.getenv("OLAMA_MODEL"))
    yield
    del ollama_client

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://chainsentinel.mihneahututui.eu",
    "https://chainsentinel.app.genez.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(text: str):
   global context
   res = ollama_client.generate(text, context)
   if context == []:
       context=res["context"]
   return {"message": res["response"]}

if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", "8000"))