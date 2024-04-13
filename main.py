from contextlib import asynccontextmanager
from http.client import HTTPException
from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import SeleniumURLLoader
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from utils import OllamaClient, ChromaClient, Chunker

from dotenv import load_dotenv

import os

load_dotenv()

ollama_client = None
ingester = None
context = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_client, ingester, ollama_client
    ollama_client = OllamaClient(os.getenv("OLLAMA_URL"), os.getenv("OLAMA_MODEL"))
    chroma_client = ChromaClient(os.getenv("CHROMA_PATH", "./chromadb")).chroma_client
    ingester = Chunker((os.getenv("CHUNKER_MODEL_NAME")), 768)
    yield
    del chroma_client, ollama_client


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

@app.post("/ingest_documents")
async def ingest_documents(files: List[UploadFile]):
    for file in files:
        data = await file.read()
        file_type = None
        if file.filename.split(".")[-1] == "pdf" or file.filename.split(".")[-1] == "PDF":
            file_type = "PDF"
        else:
            raise HTTPException(status_code=406, detail="We do not support other file types other than PDFs")
        ingester.ingest(data, file.filename, file_type, os.getenv("OLLAMA_URL"), os.getenv("E_MODEL_NAME"),os.getenv("CHROMA_PATH", "./chromadb"), os.getenv("CHROMA_COLLECTION"))
        try:
            return {"message": "ok"}
        except:
             raise HTTPException(status_code=500, detail="Problem")

@app.post("/ingest_urls")
async def ingest_urls(urls: List[str]):
    for url in urls:
        file_type = "URL"
        ingester.ingest(SeleniumURLLoader(urls=[url]).load()[0].page_content, url, file_type, os.getenv("OLLAMA_URL"), os.getenv("E_MODEL_NAME"),os.getenv("CHROMA_PATH", "./chromadb"), os.getenv("CHROMA_COLLECTION"))
        try:
            return {"message": "ok"}
        except:
             raise HTTPException(status_code=500, detail="Problem")

if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", "8000"))