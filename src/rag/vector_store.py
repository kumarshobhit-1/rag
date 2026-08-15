import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.llm_factory import LLMFactory

logger = get_logger()

class VectorStoreManager:
    """Manages persistent Chroma vector store operations."""
    
    def __init__(
        self,
        persist_dir: str = settings.CHROMA_PERSIST_DIR,
        collection_name: str = settings.COLLECTION_NAME,
        provider: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function or LLMFactory.get_embeddings(provider)
        self._vectorstore: Optional[Chroma] = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        """Initializes or loads the Chroma vector store."""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            logger.info(f"Chroma DB initialized at '{self.persist_dir}' with collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error initializing Chroma DB: {e}")
            raise RuntimeError(f"Chroma initialization error: {e}")

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._init_vectorstore()
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Adds document chunks to Chroma vector store."""
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return []
            
        try:
            ids = self.vectorstore.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} document chunks to Chroma DB.")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise RuntimeError(f"Failed to index documents: {e}")

    def search_similarity(self, query: str, k: int = settings.RETRIEVAL_K) -> List[Document]:
        """Performs basic similarity search with local fallback."""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.warning(f"Similarity search API error: {e}. Switching to local HuggingFace embeddings fallback.")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            fallback_embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self._vectorstore._embedding_function = fallback_embed
            return self.vectorstore.similarity_search(query, k=k)

    def search_mmr(
        self,
        query: str,
        k: int = settings.RETRIEVAL_K,
        fetch_k: int = settings.RETRIEVAL_FETCH_K,
        lambda_mult: float = settings.RETRIEVAL_LAMBDA_MULT
    ) -> List[Document]:
        """Performs Maximal Marginal Relevance (MMR) search for diversity."""
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            )
            return retriever.invoke(query)
        except Exception as e:
            logger.warning(f"MMR search fallback to similarity search due to: {e}")
            return self.search_similarity(query, k=k)

    def get_retriever(
        self,
        k: int = settings.RETRIEVAL_K,
        fetch_k: int = settings.RETRIEVAL_FETCH_K,
        lambda_mult: float = settings.RETRIEVAL_LAMBDA_MULT
    ):
        """Returns a LangChain retriever interface."""
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )

    def list_sources(self) -> List[Dict[str, Any]]:
        """Extracts unique source document names and chunk counts."""
        try:
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            sources: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                source_name = meta.get("source_name", meta.get("source", "Unknown Source"))
                if source_name not in sources:
                    sources[source_name] = {
                        "source_name": source_name,
                        "file_type": meta.get("file_type", "unknown"),
                        "chunk_count": 0,
                        "processed_at": meta.get("processed_at", "")
                    }
                sources[source_name]["chunk_count"] += 1
                
            return list(sources.values())
        except Exception as e:
            logger.error(f"Error listing document sources: {e}")
            return []

    def delete_source(self, source_name: str) -> bool:
        """Deletes all chunks belonging to a specific source document."""
        try:
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])
            ids_to_delete = []
            
            for doc_id, meta in zip(results["ids"], results["metadatas"]):
                if meta and (meta.get("source_name") == source_name or meta.get("source") == source_name):
                    ids_to_delete.append(doc_id)
                    
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for source: {source_name}")
                return True
            else:
                logger.warning(f"No chunks found for source: {source_name}")
                return False
        except Exception as e:
            logger.error(f"Error deleting source {source_name}: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clears all vectors in the collection."""
        try:
            collection = self.vectorstore._collection
            results = collection.get()
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info("Cleared all documents from Chroma collection.")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection size and stats."""
        try:
            count = self.vectorstore._collection.count()
            sources = self.list_sources()
            return {
                "total_chunks": count,
                "total_documents": len(sources),
                "persist_directory": self.persist_dir,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_chunks": 0, "total_documents": 0, "error": str(e)}

    def close(self):
        """Closes the vector store connection."""
        if self._vectorstore:
            try:
                if hasattr(self._vectorstore, "_client") and hasattr(self._vectorstore._client, "close"):
                    self._vectorstore._client.close()
            except Exception as e:
                logger.warning(f"Error closing Chroma client: {e}")
            self._vectorstore = None
