from typing import List
from pydantic import BaseModel
from datetime import datetime

class Metadata(BaseModel):
    source: str
    tag: str

class Sources(BaseModel):
    page_content: str
    metadata: Metadata
    type: str

class LLMResponse(BaseModel):
    query: str
    result: str
    source_documents: List[Sources] = []
