import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
        
        # API Keys
        MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
        GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
        TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
        
        # Storage Settings
        CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma-db")
        COLLECTION_NAME: str = "rag_documents"
        
        # RAG Parameters
        DEFAULT_LLM_PROVIDER: str = "mistral"
        DEFAULT_MODEL_NAME: str = "mistral-small-2506"
        CHUNK_SIZE: int = 1000
        CHUNK_OVERLAP: int = 200
        RETRIEVAL_K: int = 4
        RETRIEVAL_FETCH_K: int = 10
        RETRIEVAL_LAMBDA_MULT: float = 0.5
        
        # Server Settings
        API_HOST: str = "0.0.0.0"
        API_PORT: int = 8000
        DEBUG: bool = False

except ImportError:
    from pydantic import BaseModel
    
    class Settings(BaseModel):
        MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
        GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
        TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
        
        CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma-db")
        COLLECTION_NAME: str = "rag_documents"
        
        DEFAULT_LLM_PROVIDER: str = "mistral"
        DEFAULT_MODEL_NAME: str = "mistral-small-2506"
        CHUNK_SIZE: int = 1000
        CHUNK_OVERLAP: int = 200
        RETRIEVAL_K: int = 4
        RETRIEVAL_FETCH_K: int = 10
        RETRIEVAL_LAMBDA_MULT: float = 0.5
        
        API_HOST: str = "0.0.0.0"
        API_PORT: int = 8000
        DEBUG: bool = False

settings = Settings()
