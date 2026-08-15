import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

class DocumentProcessor:
    """Handles document loading, splitting, and metadata enrichment."""
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_file(self, file_path: str, original_filename: Optional[str] = None) -> List[Document]:
        """Loads a document based on its file extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = path.suffix.lower()
        filename = original_filename or path.name
        
        logger.info(f"Loading file: {filename} ({ext})")
        
        try:
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(path))
                docs = loader.load()
            elif ext in [".txt", ".log", ".md", ".markdown"]:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(path), encoding="utf-8")
                docs = loader.load()
            else:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(path), encoding="utf-8")
                docs = loader.load()
                
            # Enrich document metadata
            for i, doc in enumerate(docs):
                doc.metadata["source_name"] = filename
                doc.metadata["file_type"] = ext
                doc.metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i + 1
                    
            return docs
            
        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}")
            raise RuntimeError(f"Failed to load document {filename}: {str(e)}")

    def process_and_split(self, docs: List[Document]) -> List[Document]:
        """Splits loaded documents into chunks and adds chunk metadata."""
        chunks = self.text_splitter.split_documents(docs)
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["total_chunks"] = len(chunks)
            
        logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
        return chunks

    def process_file(self, file_path: str, original_filename: Optional[str] = None) -> List[Document]:
        """Helper to load and split a file in one step."""
        docs = self.load_file(file_path, original_filename)
        return self.process_and_split(docs)
