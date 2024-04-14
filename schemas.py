from typing import List
from pydantic import BaseModel
from datetime import datetime

class Chat(BaseModel):
    query: str

class IngestUrls(BaseModel):
    urls: List[str]

class Metadata(BaseModel):
    source: str

class Sources(BaseModel):
    metadata: Metadata

class LLMResponse(BaseModel):
    query: str
    result: str
    source_documents: List[Sources] = []
