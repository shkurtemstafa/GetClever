

"""Main Streamlit application for GetClever RAG System."""

import streamlit as st
import os
import sys
from pathlib import Path
import plotly.express as px
import pandas as pd
import zipfile
import requests
import tempfile
import requests
import tempfile

# Add parent directory to path to import RAG modules
sys.path.append(str(Path(__file__).parent.parent))

from rag.prompting import RAGSystem

# Vector store download URL - you'll need to replace this with your actual URL
VECTOR_STORE_URL = "https://drive.google.com/uc?export=download&id=1_g8GO7pdODTyuxGyAYg6pB2FY3Z8iLoG"

def download_vector_store():
    """Download vector store from external URL."""
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    zip_path = Path(__file__).parent.parent / "data" / "chroma_db.zip"
    
    if db_path.exists():
        return True
        
    try:
        st.info("🔄 Downloading vector store for first use...")
        
        # Create data directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the zip file
        response = requests.get(VECTOR_STORE_URL, stream=True)
        response.raise_for_status()
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_zip_path = tmp_file.name
        
        # Extract the zip file
        st.info("📦 Extracting vector store...")
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(Path(__file__).parent.parent / "data")
        
        # Clean up temporary file
        os.unlink(tmp_zip_path)
        
        st.success("✅ Vector store ready!")
        return True
        
    except Exception as e:
        st.error(f"Failed to download vector store: {str(e)}")
        return False

def ensure_vector_store_exists():
    """Ensure vector store exists by downloading if needed."""
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    zip_path = Path(__file__).parent.parent / "data" / "chroma_db.zip"
    
    # If vector store doesn't exist, try to download it
    if not db_path.exists():
        if zip_path.exists():
            # Local zip exists, extract it
            st.info("🔄 Setting up vector store for first use...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(Path(__file__).parent.parent / "data")
                st.success("✅ Vector store ready!")
                return True
            except Exception as e:
                st.error(f"Failed to extract vector store: {str(e)}")
                return False
        else:
            # No local zip, download from external source
            return download_vector_store()
    
    return db_path.exists()

# Page configuration
icon_path = str(Path(__file__).parent.parent / "dataset" / "GetClever.png")
st.set_page_config(
    page_title="GetClever",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling

# Custom CSS for professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* =====================
   GLOBAL BASE
===================== */
html, body, .stApp {
    background-color: #F4F3F0;
    font-family: 'Inter', sans-serif;
    color: #1E1E1E; ,
    font-size: 11px;   /* GLOBAL SMALL FONT */
}

/* Main container */
.stMainBlockContainer {
    background-color: #F4F3F0;
    padding-top: 1.5rem;
}

/* =====================
   HEADERS (SMALLER)
===================== */
h1 { font-size: 22px !important; font-weight: 600; }
h2 { font-size: 18px !important; font-weight: 600; }
h3 { font-size: 16px !important; font-weight: 500; }
h4 { font-size: 14.5px !important; font-weight: 500; }

/* Markdown text */
.stMarkdown, p, li, span {
    font-size: 12px !important;
    line-height: 1.5;
}

/* =====================
   SIDEBAR
===================== */
.stSidebar {
    background-color: #F4F3F0 !important;
}

.stSidebar * {
    font-size: 10.5px !important;
    color: #143d33 !important;
}

.stSidebar h1, 
.stSidebar h2, 
.stSidebar h3 {
    font-size: 12px !important;
    color: #1E1E1E !important;
}

/* =====================
   BUTTONS - PROPER CONTRAST & BIGGER TEXT
===================== */
.stButton > button {
    font-size: 13px !important;  /* Bigger button text */
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;  /* More padding */
    font-weight: 600 !important;
    border: 2px solid #143d33 !important;
    transition: all 0.2s ease !important;
}

/* PRIMARY BUTTONS: Green background = White text */
.stButton > button[kind="primary"] {
    background-color: #143d33 !important;  /* Green background */
    color: #FFFFFF !important;              /* WHITE text */
    border: 2px solid #143d33 !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #0F2A1F !important;  /* Darker green background */
    color: #FFFFFF !important;              /* WHITE text */
    border: 2px solid #0F2A1F !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(15, 42, 31, 0.3) !important;
}

/* SECONDARY BUTTONS: White background = Green text */
.stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;  /* WHITE background */
    color: #143d33 !important;              /* Green text */
    border: 2px solid #143d33 !important;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #E8E6E1 !important;  /* Light background */
    color: #0F2A1F !important;              /* Dark green text */
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(15, 42, 31, 0.3) !important;
}

/* =====================
   CHAT
===================== */
.stChatMessage {
    background-color: #FFFFFF !important;
    border: 1px solid rgba(20,61,51,0.35) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}

.stChatMessage * {
    font-size: 11px !important;
}

.stChatInput textarea {
    font-size: 11px !important;
    border-radius: 10px !important;
    border: 1px solid #143d33 !important;
}

/* =====================
   METRICS
===================== */
.stMetric {
    background-color: #FFFFFF !important;
    padding: 0.75rem !important;
    border-radius: 10px !important;
    border: 1px solid rgba(20,61,51,0.35) !important;
}

.stMetric label {
    font-size: 10px !important;
    color: #143d33 !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* ====================
   CARDS / CONTAINERS
===================== */
.main-card,
.chat-container {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    border: 1px solid rgba(20,61,51,0.35) !important;
}

/* =====================
   APP TITLE
===================== */
.app-title {
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-bottom: 0.25rem !important;
}

.app-subtitle {
    font-size: 12px !important;
    color: #143d33 !important;
}

/* =====================
   CLEAN STREAMLIT
===================== */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { visibility: hidden; }




<style>
/* Follow-up buttons */
div.stButton > button {
    background-color: #f8f9fa;
    color: #333;
    border-radius: 10px;
    border: 1px solid #ddd;
    padding: 8px 14px;
    margin: 4px 0;
    transition: all 0.2s ease-in-out;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

/* Hover effect - Light background with dark text */
div.stButton > button:hover {
    box-shadow: 0px 6px 16px rgba(15, 42, 31, 0.3) !important;
    transform: translateY(-2px) !important;
    background-color: #E8E6E1 !important;  /* Lighter version of #F4F3F0 */
    color: #0F2A1F !important;              /* Dark green text */
    border: 2px solid #143d33 !important;
}

/* Click (active) effect - Even lighter background */
div.stButton > button:active {
    box-shadow: inset 0px 3px 8px rgba(15, 42, 31, 0.4) !important;
    transform: translateY(1px) !important;
    background-color: #DDD9D2 !important;  /* Even lighter background */
    color: #0F2A1F !important;              /* Dark green text */
}


</style>
""", unsafe_allow_html=True)


# Initialize session state
if "rag_system" not in st.session_state:
    # Ensure vector store is available (download if needed)
    if ensure_vector_store_exists():
        try:
            st.session_state.rag_system = RAGSystem()
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            st.stop()
    else:
        st.error("Vector store not available. Please check deployment.")
        st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_ingested" not in st.session_state:
    # Check if documents are already ingested by checking vector store
    try:
        stats = st.session_state.rag_system.get_system_stats()
        st.session_state.documents_ingested = stats["vector_store_stats"]["count"] > 0
    except Exception as e:
        st.warning(f"Could not check vector store status: {str(e)}")
        st.session_state.documents_ingested = False


def process_followup_question(question):
    """Process a follow-up question and add both question and answer to chat history."""
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    
    # Generate response using RAG system
    response = st.session_state.rag_system.query(
        question,
        use_hybrid_search=getattr(st.session_state, 'use_hybrid', True),
        use_reranking=getattr(st.session_state, 'use_reranking', True),
        k=getattr(st.session_state, 'k_docs', 5),
        include_conversation_context=True
    )
    
    # Add assistant message to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response["answer"],
        "citations": response.get("citations", []),
        "metadata": {
            "confidence": response.get("confidence", "medium"),
            "sources_used": response.get("sources_used", 0),
            "search_method": response.get("search_method", "semantic"),
            "retrieved_docs": response.get("retrieved_docs", 0)
        },
        "followup_questions": response.get("followup_questions", [])
    })
    
    # Rerun to display the new messages
    st.rerun()


def is_no_answer_response(answer_text):
    """Check if the response is a 'no answer' type response."""
    response_lower = answer_text.lower().strip()
    
    # User-friendly no-answer patterns (no mention of documents)
    no_answer_patterns = [
        "i don't have enough information",
        "i don't currently have the necessary details",
        "there isn't enough reliable information",
        "i'm unable to find a clear answer",
        "i cannot find information",
        "no information is available",
        "insufficient information",
        "not enough information",
        "unable to provide",
        "not available at this time"
    ]
    
    # Check if response contains any no-answer patterns
    for pattern in no_answer_patterns:
        if pattern in response_lower:
            return True
    
    # Additional check: very short responses that are likely "no answer"
    if len(response_lower) < 100 and any(word in response_lower for word in ["don't", "can't", "cannot", "unable", "insufficient"]):
        return True
        
    return False


def main():
    """Main application function."""
    # Enhanced logo design for GetClever
    st.markdown("""
    <div class="app-header" style="
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #143d33 0%, #1a4d3d 100%);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(20, 61, 51, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
    ">
        <h1 style="
            font-size: 56px;
            font-weight: 900;
            margin: 0;
            background: linear-gradient(45deg, #F4F3F0, #ffffff, #e8e6e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: 2px;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        ">GetClever</h1>
        <div style="
            width: 80px;
            height: 3px;
            background: linear-gradient(90deg, #F4F3F0, #143d33, #F4F3F0);
            margin: 10px auto;
            border-radius: 2px;
        "></div>
        <p style="
            font-size: 18px;
            font-weight: 500;
            color: #F4F3F0;
            margin: 10px 0 0 0;
            opacity: 0.9;
            letter-spacing: 1px;
        ">Intelligent Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)


    # Main chat interface (full width)
    if not st.session_state.documents_ingested:
        vector_store_path = Path(__file__).parent.parent / "data" / "chroma_db"
        dataset_path = Path(__file__).parent.parent / "dataset"
        
        st.markdown("""
        <div class="main-card">
            <h3>🚀 Getting Started with GetClever</h3>
        """, unsafe_allow_html=True)
        
        if vector_store_path.exists():
            st.markdown("""
            <p><strong>✅ Pre-built vector store detected!</strong></p>
            <p>Click <strong>"Load Vector Store"</strong> in the sidebar to start chatting immediately.</p>
            """, unsafe_allow_html=True)
        elif dataset_path.exists() and any(dataset_path.iterdir()):
            st.markdown("""
            <p><strong>📁 Dataset folder found!</strong></p>
            <p>Click <strong>"Ingest Documents"</strong> in the sidebar to process your documents.</p>
            <ul>
                <li><strong>Supported formats:</strong> PDF, DOCX, Markdown (.md), Text (.txt)</li>
                <li><strong>Processing time:</strong> ~2-5 minutes depending on document size</li>
                <li><strong>One-time setup:</strong> After ingestion, you can start asking questions!</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p><strong>⚠️ No documents or vector store found</strong></p>
            <p>This appears to be a fresh deployment. You'll need to:</p>
            <ul>
                <li>Upload documents to the <code>dataset/</code> folder, or</li>
                <li>Deploy with a pre-built vector store</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        return
    else:
        # Show ready status
        stats = st.session_state.rag_system.get_system_stats()
        st.markdown(f"""
            <div style="
               background-color: #143d33;       /* Dark green background */
               color: #F4F3F0;                  /* Off-white text */
               font-weight: bold;                /* Bold text */
               font-size: 12px;                  /* Text size */
               padding: 12px 20px;               /* Padding around text */
               border-radius: 10px;              /* Rounded corners */
               box-shadow: 3px 3px 8px rgba(0,0,0,0.4); /* Subtle shadow */
            ">
                Ready! {stats['vector_store_stats']['count']} document chunks loaded
            </div>
            """, unsafe_allow_html=True)

    # Chat container
    st.markdown("### Chat with your Documents")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Check if this is a "no answer" response
                is_no_answer = message.get("has_substantive_answer", True) == False or is_no_answer_response(message["content"])
                
                # Only show citations and metadata if there's an actual answer
                if not is_no_answer:
                    # Show citations if available
                    if message.get("citations"):
                        with st.expander("📚 Citations"):
                            for citation in message["citations"]:
                                st.write(f"• {citation}")
                    
                    # Show confidence and stats
                    if message.get("metadata"):
                        metadata = message["metadata"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            confidence = metadata.get("confidence", "medium")
                            st.markdown(f"**Evidence Strength:** {confidence.title()}")
                        with col2:
                            st.write(f"**Sources:** {metadata.get('sources_used', 0)}")
                        with col3:
                            method = metadata.get('search_method', 'semantic')
                            st.write(f"**Search:** {method.title()}")
                
                # Always show follow-up questions (even for no-answer responses)
                if message.get("followup_questions"):
                    st.markdown("** Follow-up questions:**")
                    for j, question in enumerate(message["followup_questions"]):
                        if st.button(f"{question}", key=f"followup_{i}_{j}"):
                            # Process the follow-up question immediately
                            process_followup_question(question)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_system.query(
                    prompt,
                    use_hybrid_search=True,
                    use_reranking=True,
                    k=5,
                    include_conversation_context=True
                )
            
            # Display answer
            st.write(response["answer"])
            
            # Check if this is a "no answer" response
            is_no_answer = response.get("has_substantive_answer", True) == False or is_no_answer_response(response["answer"])
            
            # Only show citations and metadata if there's an actual answer
            if not is_no_answer:
                # Show citations
                if response.get("citations"):
                    with st.expander("📚 Citations"):
                        for citation in response["citations"]:
                            st.write(f"• {citation}")
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    confidence = response.get("confidence", "medium")
                    st.markdown(f"**Evidence Strength:** {confidence.title()}")
                with col2:
                    st.write(f"**Sources:** {response.get('sources_used', 0)}")
                with col3:
                    method = response.get('search_method', 'semantic')
                    st.write(f"**Search:** {method.title()}")
            
            # Always show follow-up questions (even for no-answer responses)
            if response.get("followup_questions"):
                st.markdown("** Follow-up questions:**")
                for j, question in enumerate(response["followup_questions"]):
                    if st.button(f"{question}", key=f"new_followup_{j}"):
                        # Process the follow-up question immediately
                        process_followup_question(question)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "citations": response.get("citations", []),
                "metadata": {
                    "confidence": response.get("confidence", "medium"),
                    "sources_used": response.get("sources_used", 0),
                    "search_method": response.get("search_method", "semantic"),
                    "retrieved_docs": response.get("retrieved_docs", 0)
                },
                "followup_questions": response.get("followup_questions", [])
            })
    
    st.markdown('</div>', unsafe_allow_html=True)


def show_analytics_page():
    """Show analytics and observability dashboard."""
    st.markdown("### Analytics Dashboard")
    
    stats = st.session_state.rag_system.get_system_stats()
    
    # Overview metrics
    st.markdown("#### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", stats["system_stats"]["total_queries"])
    with col2:
        st.metric("Successful Answers", stats["system_stats"]["successful_answers"])
    with col3:
        st.metric("Success Rate", stats["system_stats"]["success_rate"])
    with col4:
        st.metric("Documents", stats["vector_store_stats"]["count"])
    
    # Document sources
    st.markdown("#### Document Sources")
    sources = st.session_state.rag_system.get_document_sources()
    
    if sources:
        df_sources = pd.DataFrame(sources)
        
        fig_pie = px.pie(
            df_sources,
            values='chunks',
            names='name',
            title="Document Distribution by Source",
            color_discrete_sequence=['#143d33', '#1E1E1E', '#F4F3F0']
        )
        fig_pie.update_layout(
            plot_bgcolor='#F4F3F0', 
            paper_bgcolor='#F4F3F0', 
            font_color='#1E1E1E'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        fig_bar = px.bar(
            df_sources.head(10),
            x='name',
            y='chunks',
            title="Top 10 Sources by Chunk Count",
            color_discrete_sequence=['#143d33']
        )
        fig_bar.update_xaxes(tickangle=45)
        fig_bar.update_layout(
            plot_bgcolor='#F4F3F0', 
            paper_bgcolor='#F4F3F0', 
            font_color='#1E1E1E'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("#### All Sources")
        st.dataframe(df_sources, use_container_width=True)
    
    # Chat history analysis
    if st.session_state.chat_history:
        st.markdown("#### Chat Analysis")
        
        confidences = [
            msg["metadata"].get("confidence","medium")
            for msg in st.session_state.chat_history
            if msg["role"]=="assistant" and msg.get("metadata")
        ]
        
        if confidences:
            confidence_counts = pd.Series(confidences).value_counts()
            
            fig_confidence = px.bar(
                x=confidence_counts.index,
                y=confidence_counts.values,
                title="Answer Confidence Distribution",
                labels={'x':'Confidence Level','y':'Count'},
                color_discrete_sequence=['#143d33']
            )
            fig_confidence.update_layout(
                plot_bgcolor='#F4F3F0', 
                paper_bgcolor='#F4F3F0', 
                font_color='#1E1E1E'
            )
            st.plotly_chart(fig_confidence, use_container_width=True)


if __name__ == "__main__":
    # Sidebar navigation and controls
    with st.sidebar:
        # Navigation
        page = st.selectbox("Navigate", ["Chat", "Analytics"], index=0)
        
        st.markdown("---")
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY environment variable.")
            st.stop()
        
        # Document ingestion section
        st.markdown("#### Document Management")
        
        # Check if vector store exists
        vector_store_path = Path(__file__).parent.parent / "data" / "chroma_db"
        
        if st.session_state.documents_ingested:
            st.success("✅ Documents already loaded!")
            stats = st.session_state.rag_system.get_system_stats()
            st.info(f"Ready with {stats['vector_store_stats']['count']} document chunks")
        elif vector_store_path.exists():
            st.info("📦 Pre-built vector store detected")
            if st.button("🔄 Load Vector Store", type="primary", use_container_width=True):
                try:
                    # Force reload the RAG system to pick up existing vector store
                    st.session_state.rag_system = RAGSystem()
                    stats = st.session_state.rag_system.get_system_stats()
                    if stats["vector_store_stats"]["count"] > 0:
                        st.session_state.documents_ingested = True
                        st.success(f"✅ Loaded {stats['vector_store_stats']['count']} document chunks!")
                        st.rerun()
                    else:
                        st.error("Vector store exists but appears empty")
                except Exception as e:
                    st.error(f"Failed to load vector store: {str(e)}")
        else:
            st.warning("⚠️ No vector store found")
            
            # Check if dataset exists for ingestion
            dataset_path = Path(__file__).parent.parent / "dataset"
            if dataset_path.exists() and any(dataset_path.iterdir()):
                st.info("📁 Dataset folder found - you can ingest documents")
            else:
                st.error("❌ No dataset folder found for ingestion")
        
        # Always show ingest button (for re-ingestion or first-time ingestion)
        if st.button("📥 Ingest Documents", type="secondary", use_container_width=True, key="ingest_btn"):
            dataset_path = Path(__file__).parent.parent / "dataset"
            if not dataset_path.exists() or not any(dataset_path.iterdir()):
                st.error("❌ Dataset folder not found or empty. Cannot ingest documents.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        result = st.session_state.rag_system.ingest_documents()
                        if result["success"]:
                            st.success(result["message"])
                            st.session_state.documents_ingested = True
                            # Show ingestion stats
                            if "stats" in result:
                                stats = result["stats"]
                                if "document_stats" in stats:
                                    doc_stats = stats["document_stats"]
                                    st.markdown(f"**Total chunks:** {doc_stats.get('total_documents', 0)}")
                                    st.markdown(f"**Sources:** {len(doc_stats.get('sources', []))}")
                            st.rerun()
                        else:
                            st.error(result["message"])
                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")
        
        # System stats
        st.markdown("####  System Statistics")
        stats = st.session_state.rag_system.get_system_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats["system_stats"]["total_queries"])
            st.metric("Success Rate", stats["system_stats"]["success_rate"])
        
        with col2:
            st.metric("Documents", stats["vector_store_stats"]["count"])
            st.metric("Sources", stats["sources"])
        
        # Document sources
        if st.session_state.documents_ingested:
            st.markdown("#### 📋 Document Sources")
            sources = st.session_state.rag_system.get_document_sources()
            if sources:
                for source in sources[:5]:  # Show top 5 sources
                    with st.expander(f"📄 {source['name']} ({source['chunks']} chunks)"):
                        st.write(f"**Type:** {source['type']}")
                        if source.get('total_pages'):
                            st.write(f"**Pages:** {source['total_pages']}")
        
        # System controls
        st.markdown("####  System Controls")
        
        # Advanced settings for chat
        if page == "Chat":
            use_hybrid = st.checkbox("Use Hybrid Search", value=True)
            use_reranking = st.checkbox("Use Reranking", value=True)
            k_docs = st.slider("Max Documents", 3, 10, 5)
            
            # Store in session state for use in main
            st.session_state.use_hybrid = use_hybrid
            st.session_state.use_reranking = use_reranking
            st.session_state.k_docs = k_docs
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.rag_system.clear_conversation_history()
                st.rerun()
        
        with col2:
            if st.button(" Reset System", use_container_width=True):
                if st.session_state.rag_system.reset_system():
                    st.session_state.documents_ingested = False
                    st.session_state.chat_history = []
                    st.success("System reset!")
                    st.rerun()
    
    # Main content based on navigation
    if page == "Chat":
        main()
    elif page == "Analytics":
        show_analytics_page()