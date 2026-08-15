import os
import gc
import tempfile
import pytest
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStoreManager
from src.rag.rag_chain import RAGPipeline

def test_document_processor():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as tmp:
        tmp.write("Artificial Intelligence and Machine Learning are transforming modern industry.\n" * 5)
        tmp_path = tmp.name
        
    try:
        chunks = processor.process_file(tmp_path, original_filename="test_doc.txt")
        assert len(chunks) > 0
        assert chunks[0].metadata["source_name"] == "test_doc.txt"
        assert "chunk_id" in chunks[0].metadata
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_vector_store_manager():
    fake_embeddings = FakeEmbeddings(size=384)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vm = VectorStoreManager(
            persist_dir=tmpdir,
            collection_name="test_collection",
            embedding_function=fake_embeddings
        )
        
        try:
            docs = [
                Document(page_content="Deep Learning uses multi-layer neural networks.", metadata={"source_name": "dl.txt", "page": 1}),
                Document(page_content="Natural Language Processing handles text processing.", metadata={"source_name": "nlp.txt", "page": 1})
            ]
            
            ids = vm.add_documents(docs)
            assert len(ids) == 2
            
            sources = vm.list_sources()
            source_names = [s["source_name"] for s in sources]
            assert "dl.txt" in source_names
            assert "nlp.txt" in source_names
            
            results = vm.search_similarity("neural networks", k=1)
            assert len(results) == 1
            assert results[0].page_content is not None
            
            deleted = vm.delete_source("dl.txt")
            assert deleted is True
            
            sources_after = vm.list_sources()
            assert len(sources_after) == 1
        finally:
            vm.close()
            gc.collect()

def test_rag_citations_formatting():
    fake_embeddings = FakeEmbeddings(size=384)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vm = VectorStoreManager(
            persist_dir=tmpdir,
            collection_name="test_collection_citations",
            embedding_function=fake_embeddings
        )
        
        try:
            pipeline = RAGPipeline(vector_store_manager=vm)
            docs = [
                Document(page_content="FastAPI is a modern web framework.", metadata={"source_name": "api.pdf", "page": 2, "chunk_id": 0})
            ]
            citations = pipeline.format_citations(docs)
            assert len(citations) == 1
            assert citations[0]["source_name"] == "api.pdf"
            assert citations[0]["page"] == 2
            assert "FastAPI" in citations[0]["snippet"]
        finally:
            vm.close()
            gc.collect()
