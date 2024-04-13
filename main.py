from contextlib import asynccontextmanager
from fastapi import FastAPI

from utils import OllamaClient

from dotenv import load_dotenv

import os

load_dotenv()

ollama_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_client
    ollama_client = OllamaClient(os.getenv("OLLAMA_URL"), os.getenv("OLAMA_MODEL"))
    yield
    del ollama_client

app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(text: str):
   global context
   res = ollama_client.generate(text, context)
   if context == []:
       context=res["context"]
   return res["response"]

if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", "8000"))