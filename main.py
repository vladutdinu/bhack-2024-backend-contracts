from contextlib import asynccontextmanager
from http.client import HTTPException
from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import SeleniumURLLoader
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from utils import OllamaClient, ChromaClient, Chunker, RAGPipeline
from schemas import IngestUrls, LLMResponse, Chat, Metadata, Sources
from dotenv import load_dotenv

import os

load_dotenv()

pipeline = None
chroma_client = None
ingester = None
ollama_clien = None
context = []
rejected_file_types = ["pdf", "docx", "PDF", "DOCX"]
allowed_file_types = ["sol"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, chroma_client, ingester, ollama_client
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    ollama_client = OllamaClient(os.getenv("OLLAMA_URL"), os.getenv("OLLAMA_MODEL"))
    chroma_client = ChromaClient(os.getenv("CHROMA_PATH", "./chromadb")).chroma_client
    pipeline = RAGPipeline(
        os.getenv("RAG_MODEL_NAME"),
        model_kwargs,
        os.getenv("OLLAMA_URL"),
        os.getenv("OLLAMA_MODEL"),
        chroma_client,
        os.getenv("CHROMA_COLLECTION"),
    )
    ingester = Chunker((os.getenv("CHUNKER_MODEL_NAME")), 768)
    yield
    del chroma_client, pipeline, ollama_client, pipeline, ingester


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://chainsentinel.mihneahututui.eu",
    "https://chainsentinel.app.genez.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(query: Chat) -> LLMResponse:
    global context
    res = ollama_client.generate(query.query, context)
    if context == []:
        context = res["context"]
    llm_response = LLMResponse(query=query.query, result=res["response"])
    return llm_response


@app.post("/verify_contract")
async def verify_contract(file: UploadFile) -> LLMResponse:
    data = await file.read()
    if file.filename.split(".")[-1] in rejected_file_types:
        raise HTTPException(
            status_code=406,
            detail="We do not support other file types other than {}".format(
                "".join(allowed_file_types)
            ),
        )
    else:
        res = pipeline.invoke(data)
        print(res.keys())
        print(res['source_documents'])
        return LLMResponse(
            query=res["query"],
            result=res["result"],
            source_documents=[
                Sources(
                    page_content=src.page_content,
                    metadata=Metadata(
                        source=src.metadata["source"], tag=src.metadata["tag"]
                    ),
                )
                for src in res["source_documents"]
            ],
        )


@app.post("/verify_text")
async def verify_text(query: Chat) -> LLMResponse:
    res = pipeline.invoke(query.query)
    return LLMResponse(
        query=res["query"],
        result=res["result"],
        source_documents=[
            Sources(
                page_content=src.page_content,
                metadata=Metadata(
                    source=src.metadata["source"], tag=src.metadata["tag"]
                ),
            )
            for src in res["source_documents"]
        ],
    )


@app.post("/ingest_documents")
async def ingest_documents(file: UploadFile):
    data = await file.read()
    file_type = None
    if file.filename.split(".")[-1] == "pdf" or file.filename.split(".")[-1] == "PDF":
        file_type = "PDF"
    else:
        raise HTTPException(
            status_code=406, detail="We do not support other file types other than PDFs"
        )
    try:
        ingester.ingest(
            data,
            file.filename,
            file_type,
            os.getenv("OLLAMA_URL"),
            os.getenv("E_MODEL_NAME"),
            os.getenv("CHROMA_PATH", "./chromadb"),
            os.getenv("CHROMA_COLLECTION"),
        )
    except:
        raise HTTPException(status_code=500, detail="Problem")
    return {"message": "ok"}


@app.post("/ingest_urls")
async def ingest_urls(urls_to_ingest: IngestUrls):
    for url in urls_to_ingest.urls:
        file_type = "URL"
        try:
            ingester.ingest(
                SeleniumURLLoader(urls=[url]).load()[0].page_content,
                url,
                file_type,
                os.getenv("OLLAMA_URL"),
                os.getenv("E_MODEL_NAME"),
                os.getenv("CHROMA_PATH", "./chromadb"),
                os.getenv("CHROMA_COLLECTION"),
            )
        except:
            raise HTTPException(status_code=500, detail="Problem")
        return {"message": "ok"}


if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", "8000"))
