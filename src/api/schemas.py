from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    total_chunks: int
    total_documents: int
    persist_directory: str

class ChatRequest(BaseModel):
    question: str = Field(..., description="User query for RAG processing")
    llm_provider: Optional[str] = Field("mistral", description="Target LLM provider (mistral, openai, google)")
    model_name: Optional[str] = Field(None, description="Specific model name override")
    top_k: Optional[int] = Field(4, description="Number of context chunks to retrieve")

class CitationModel(BaseModel):
    citation_id: int
    source_name: str
    page: Any
    chunk_id: int
    snippet: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationModel]
    llm_provider: str

class DocumentSource(BaseModel):
    source_name: str
    file_type: str
    chunk_count: int
    processed_at: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentSource]
    total_documents: int

class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks_indexed: int
    message: str

class DeleteResponse(BaseModel):
    source_name: str
    success: bool
    message: str
