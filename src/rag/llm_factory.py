import os
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

class LLMFactory:
    """Factory for creating Embeddings and Chat LLM instances."""
    
    @staticmethod
    def get_embeddings(provider: Optional[str] = None) -> Embeddings:
        provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
        
        try:
            if provider == "mistral":
                if not settings.MISTRAL_API_KEY:
                    logger.warning("MISTRAL_API_KEY is not set. Falling back to local HuggingFace embeddings.")
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                from langchain_mistralai import MistralAIEmbeddings
                return MistralAIEmbeddings(
                    model="mistral-embed",
                    api_key=settings.MISTRAL_API_KEY
                )
            elif provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            elif provider == "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=settings.GOOGLE_API_KEY
                )
            elif provider in ["huggingface", "local"]:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            else:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error instantiating {provider} embeddings: {e}. Falling back to HuggingFace.")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    @staticmethod
    def get_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2
    ) -> BaseChatModel:
        provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
        
        if provider == "mistral":
            model = model_name or "mistral-small-2506"
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(
                model=model,
                temperature=temperature,
                api_key=settings.MISTRAL_API_KEY
            )
        elif provider == "openai":
            model = model_name or "gpt-4o-mini"
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.OPENAI_API_KEY
            )
        elif provider == "google":
            model = model_name or "gemini-1.5-flash"
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY
            )
        else:
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(
                model=model_name or "mistral-small-2506",
                temperature=temperature,
                api_key=settings.MISTRAL_API_KEY
            )
