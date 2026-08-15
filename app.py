import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStoreManager
from src.rag.rag_chain import RAGPipeline

logger = get_logger()

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for Modern Dark Glassmorphism UI
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Citation Cards */
    .citation-card {
        background: rgba(30, 41, 59, 0.4);
        border-left: 3px solid #38bdf8;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CACHED RESOURCES ----------------
@st.cache_resource
def get_vector_manager(provider: str = "mistral"):
    return VectorStoreManager(provider=provider)

@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

document_processor = get_document_processor()

# ---------------- SESSION STATE INIT ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=64)
    st.title("⚙️ RAG Engine Control")
    
    st.markdown("---")
    
    # Provider & Model Settings
    st.subheader("🤖 Model Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["mistral", "openai", "google", "huggingface"],
        index=0,
        help="Select the AI LLM provider"
    )
    
    if llm_provider == "mistral":
        default_model = "mistral-small-2506"
    elif llm_provider == "openai":
        default_model = "gpt-4o-mini"
    elif llm_provider == "google":
        default_model = "gemini-1.5-flash"
    else:
        default_model = "mistral-small-2506"
        
    model_name = st.text_input("Model Name", value=default_model)
    top_k = st.slider("Context Chunks (k)", min_value=1, max_value=10, value=4)
    
    vector_manager = get_vector_manager(provider=llm_provider)
    
    st.markdown("---")
    
    # Document Upload Section
    st.subheader("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing & indexing documents..."):
                total_chunks_added = 0
                for file in uploaded_files:
                    suffix = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    try:
                        chunks = document_processor.process_file(tmp_path, original_filename=file.name)
                        indexed = vector_manager.add_documents(chunks)
                        total_chunks_added += len(indexed)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                st.success(f"Indexed {total_chunks_added} chunks across {len(uploaded_files)} files!")
                st.rerun()
                
    st.markdown("---")
    
    # Knowledge Base Stats & Management
    st.subheader("📊 Knowledge Base Stats")
    stats = vector_manager.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{stats.get('total_documents', 0)}</div>
            <div class="stat-lbl">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{stats.get('total_chunks', 0)}</div>
            <div class="stat-lbl">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # List ingested docs
    sources = vector_manager.list_sources()
    if sources:
        with st.expander("📚 Ingested Documents", expanded=False):
            for doc in sources:
                col_d1, col_d2 = st.columns([3, 1])
                with col_d1:
                    st.caption(f"**{doc['source_name']}** ({doc['chunk_count']} chunks)")
                with col_d2:
                    if st.button("🗑️", key=f"del_{doc['source_name']}", help="Delete document"):
                        vector_manager.delete_source(doc['source_name'])
                        st.toast(f"Deleted {doc['source_name']}")
                        st.rerun()
                        
    if st.button("🔥 Clear Entire Knowledge Base", use_container_width=True):
        if vector_manager.clear_collection():
            st.toast("Knowledge base cleared!")
            st.rerun()

# ---------------- MAIN UI HEADER ----------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">📚 RAG PDF & Document Assistant</h1>
    <div class="main-subtitle">Enterprise-grade Retrieval-Augmented Generation system with real-time vector search & citations.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- CHAT INTERFACE ----------------
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📄 View Source Context & Citations"):
                for cite in message["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>Source:</strong> {cite['source_name']} | <strong>Page:</strong> {cite['page']} | <strong>Chunk ID:</strong> {cite['chunk_id']}<br>
                        <em>"{cite['snippet']}"</em>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
query = st.chat_input("Ask any question based on your uploaded documents...")

if query:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process Query with RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating response..."):
            pipeline = RAGPipeline(
                vector_store_manager=vector_manager,
                llm_provider=llm_provider,
                model_name=model_name
            )
            
            try:
                res = pipeline.query(question=query, k=top_k)
                answer = res["answer"]
                citations = res["citations"]

                st.markdown(answer)
                
                if citations:
                    with st.expander("📄 View Source Context & Citations"):
                        for cite in citations:
                            st.markdown(f"""
                            <div class="citation-card">
                                <strong>Source:</strong> {cite['source_name']} | <strong>Page:</strong> {cite['page']} | <strong>Chunk ID:</strong> {cite['chunk_id']}<br>
                                <em>"{cite['snippet']}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                            
                # Append Assistant Message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg:
                    st.error(f"🔑 **API Key Error (401 Unauthorized)**: The `{llm_provider.upper()}_API_KEY` in your `.env` file is invalid or expired. Please update line 4 of `.env` with a valid key from https://console.mistral.ai or switch the provider in the sidebar.")
                else:
                    st.error(f"❌ Error generating response: {error_msg}")

# Option to clear chat
if st.session_state.messages:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()