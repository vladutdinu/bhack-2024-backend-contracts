from typing import List
from pydantic import BaseModel
from datetime import datetime

class Chat(BaseModel):
    query: str

class IngestUrls(BaseModel):
    urls: List[str]

class Metadata(BaseModel):
    source: str
    tag: str

class Sources(BaseModel):
    page_content: str
    metadata: Metadata

class LLMResponse(BaseModel):
    query: str
    result: str
    source_documents: List[Sources] = []
