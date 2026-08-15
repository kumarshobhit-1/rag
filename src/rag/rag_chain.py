from typing import List, Dict, Any, Generator, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.vector_store import VectorStoreManager
from src.rag.llm_factory import LLMFactory

logger = get_logger()

PROMPT_TEMPLATE = """You are an enterprise AI assistant specializing in document analysis.
Use ONLY the provided context below to answer the question accurately and concisely.

Strict Rules:
1. Base your answer strictly on the provided Context.
2. If the answer is not present in the context, respond with: "I could not find the answer in the document."
3. Do NOT make up information or use outside knowledge.
4. If relevant, cite source references where applicable.

Context:
{context}

Question:
{question}

Answer:"""

class RAGPipeline:
    """Enterprise RAG pipeline managing query execution, context retrieval, and citations."""
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.vector_store_manager = vector_store_manager or VectorStoreManager(provider=llm_provider)
        self.llm_provider = llm_provider or settings.DEFAULT_LLM_PROVIDER
        self.model_name = model_name or settings.DEFAULT_MODEL_NAME
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve_context(self, question: str, k: int = settings.RETRIEVAL_K) -> List[Document]:
        """Retrieves relevant document chunks using MMR."""
        logger.info(f"Retrieving context for query: '{question}'")
        return self.vector_store_manager.search_mmr(question, k=k)

    def format_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Formats document metadata into clean citation metadata."""
        citations = []
        for i, doc in enumerate(docs):
            citations.append({
                "citation_id": i + 1,
                "source_name": doc.metadata.get("source_name", doc.metadata.get("source", "Document")),
                "page": doc.metadata.get("page", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", i),
                "snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
            })
        return citations

    def query(
        self,
        question: str,
        k: int = settings.RETRIEVAL_K,
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes a full RAG query and returns response with citations."""
        target_provider = provider or self.llm_provider
        target_model = model_name or self.model_name
        
        # 1. Retrieve docs
        docs = self.retrieve_context(question, k=k)
        
        if not docs:
            return {
                "answer": "I could not find the answer in the document.",
                "citations": [],
                "context_used": ""
            }
            
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # 2. Build prompt
        formatted_prompt = self.prompt.format(
            context=context_str,
            question=question
        )
        
        # 3. Call LLM
        logger.info(f"Invoking LLM ({target_provider} / {target_model})")
        llm = LLMFactory.get_llm(provider=target_provider, model_name=target_model)
        
        response = llm.invoke(formatted_prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        citations = self.format_citations(docs)
        
        return {
            "answer": answer_text,
            "citations": citations,
            "context_used": context_str
        }

    def stream_query(
        self,
        question: str,
        k: int = settings.RETRIEVAL_K,
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams LLM tokens response chunk by chunk."""
        target_provider = provider or self.llm_provider
        target_model = model_name or self.model_name
        
        docs = self.retrieve_context(question, k=k)
        if not docs:
            yield "I could not find the answer in the document."
            return
            
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = self.prompt.format(
            context=context_str,
            question=question
        )
        
        llm = LLMFactory.get_llm(provider=target_provider, model_name=target_model)
        for chunk in llm.stream(formatted_prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content
