import os
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from src.core.config import settings
from src.core.logger import get_logger
from src.api.schemas import (
    HealthResponse, ChatRequest, ChatResponse,
    DocumentListResponse, UploadResponse, DeleteResponse
)
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStoreManager
from src.rag.rag_chain import RAGPipeline

logger = get_logger()

app = FastAPI(
    title="RAG Enterprise API",
    description="High-performance Production REST API for Document Ingestion & RAG Question Answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons initialization
vector_manager = VectorStoreManager()
document_processor = DocumentProcessor()
rag_pipeline = RAGPipeline(vector_store_manager=vector_manager)

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """System health check and vector database stats."""
    stats = vector_manager.get_stats()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        total_chunks=stats.get("total_chunks", 0),
        total_documents=stats.get("total_documents", 0),
        persist_directory=stats.get("persist_directory", settings.CHROMA_PERSIST_DIR)
    )

@app.post("/api/v1/documents/upload", response_model=UploadResponse, tags=["Document Management"])
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF/TXT/MD document, extracts text chunks, and indexes them into Chroma DB."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    logger.info(f"Received file upload request: {file.filename}")
    
    # Save to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        chunks = document_processor.process_file(tmp_path, original_filename=file.filename)
        indexed_ids = vector_manager.add_documents(chunks)
        
        return UploadResponse(
            filename=file.filename,
            status="success",
            chunks_indexed=len(indexed_ids),
            message=f"Successfully processed and indexed '{file.filename}' ({len(indexed_ids)} chunks)."
        )
    except Exception as e:
        logger.error(f"Upload error for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/api/v1/documents", response_model=DocumentListResponse, tags=["Document Management"])
async def list_documents():
    """Lists all ingested documents in the vector store."""
    sources = vector_manager.list_sources()
    return DocumentListResponse(
        documents=sources,
        total_documents=len(sources)
    )

@app.delete("/api/v1/documents/{source_name}", response_model=DeleteResponse, tags=["Document Management"])
async def delete_document(source_name: str):
    """Deletes all chunks of a specific document by source name."""
    success = vector_manager.delete_source(source_name)
    if success:
        return DeleteResponse(
            source_name=source_name,
            success=True,
            message=f"Successfully deleted document '{source_name}'."
        )
    else:
        raise HTTPException(status_code=404, detail=f"Document '{source_name}' not found or could not be deleted.")

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["RAG QA Engine"])
async def chat_query(request: ChatRequest):
    """Queries the RAG system and returns answer with context citations."""
    try:
        res = rag_pipeline.query(
            question=request.question,
            k=request.top_k or settings.RETRIEVAL_K,
            provider=request.llm_provider,
            model_name=request.model_name
        )
        return ChatResponse(
            question=request.question,
            answer=res["answer"],
            citations=res["citations"],
            llm_provider=request.llm_provider or settings.DEFAULT_LLM_PROVIDER
        )
    except Exception as e:
        logger.error(f"Chat query error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")

@app.post("/api/v1/chat/stream", tags=["RAG QA Engine"])
async def chat_query_stream(request: ChatRequest):
    """Streams RAG tokens response chunk by chunk."""
    try:
        def stream_generator():
            for chunk in rag_pipeline.stream_query(
                question=request.question,
                k=request.top_k or settings.RETRIEVAL_K,
                provider=request.llm_provider,
                model_name=request.model_name
            ):
                yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
