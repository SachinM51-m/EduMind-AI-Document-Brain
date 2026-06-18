import os
import sys
import json
import streamlit as st
import pandas as pd

# Add the workspace directory to the python path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_processor import process_document
from src.vector_store import LocalVectorStore
from src.llm_handler import OllamaHandler, get_qa_prompt, get_notes_prompt, get_quiz_prompt

# ----------------- PAGE STYLING & CONFIG -----------------
st.set_page_config(
    page_title="EduMind AI - Intelligent Study Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules using Outfit font, smooth gradients, and glassmorphism containers
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom CSS to inject dark-tech aesthetics */
.stApp {
    background: radial-gradient(circle at 50% 50%, #0F111A 0%, #06070B 100%);
    color: #E2E8F0;
}

/* Hide default streamlit indicators */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Custom premium container/card rules */
.glass-panel {
    background: rgba(30, 41, 59, 0.3);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.citation-card {
    background: rgba(99, 102, 241, 0.04);
    border-left: 4px solid #6366F1;
    border-radius: 4px;
    padding: 12px 18px;
    margin: 10px 0;
}

.title-gradient {
    background: linear-gradient(135deg, #A5B4FC 0%, #6366F1 50%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 5px;
}

/* General Content Contrast Adjustments */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}

p, span, li {
    color: #E2E8F0;
}

/* Custom buttons styling */
div.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5) !important;
}

/* Tabs premium styling */
button[data-baseweb="tab"] {
    color: #94A3B8 !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    transition: color 0.3s ease !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #818CF8 !important;
    font-weight: 700 !important;
    border-bottom-color: #818CF8 !important;
}

/* Sidebar Custom Styling & High-Contrast Overrides */
section[data-testid="stSidebar"] {
    background-color: #0B0C15;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Targets all text categories inside the sidebar to force highly visible contrast */
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] li {
    color: #E2E8F0 !important;
}

/* Forces headers inside the sidebar to stand out as bright white */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] strong {
    color: #FFFFFF !important;
}

/* Explicit labels on the sidebar widgets (such as uploader labels) */
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

/* File Uploader widget border, background and contrast styling */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* Force all text inside file uploader (like descriptions and small details) to be bright and clear */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #E2E8F0 !important;
}

/* Keep the click-to-upload button text inside the uploader dark for readability on light button background */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
    color: #0F111A !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDescription"] {
    color: #E2E8F0 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Maintain proper button text color within the sidebar (e.g. index/reset buttons) */
section[data-testid="stSidebar"] button p,
section[data-testid="stSidebar"] button span {
    color: #FFFFFF !important;
}

/* Help tooltip icon contrast */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {
    color: #FFFFFF !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* Index Statistics Metric visibility override */
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
    color: #A5B4FC !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
}

/* Custom indicator badges */
.status-badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.status-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981 !important;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-info {
    background-color: rgba(99, 102, 241, 0.15);
    color: #818CF8 !important;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.status-warning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B !important;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.status-error {
    background-color: rgba(239, 68, 68, 0.15);
    color: #EF4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
</style>
""", unsafe_allow_html=True)


# ----------------- INITIALIZE STATE & CLIENTS -----------------
@st.cache_resource
def get_vector_store():
    return LocalVectorStore()

@st.cache_resource
def get_ollama_handler():
    return OllamaHandler()

store = get_vector_store()
ollama_handler = get_ollama_handler()

# Initialize session state variables for chat logs, current documents, and quiz persistence
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<h2 style='color:#F3F4F6;'>🧠 EduMind Brain Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. Connection Status
    st.markdown("### 🖥️ Model Connection Status")
    is_connected, conn_msg = ollama_handler.check_connection()
    if is_connected:
        if "Cloud Fallback" in conn_msg:
            st.markdown(f'<div class="status-badge status-info">🌐 Cloud Fallback</div>', unsafe_allow_html=True)
            st.info(conn_msg)
        elif "was not found" in conn_msg:
            st.markdown(f'<div class="status-badge status-warning">⚠️ Model Missing</div>', unsafe_allow_html=True)
            st.warning("Connected to Ollama, but 'llama3.2:1b' was not found. Please run `ollama pull llama3.2:1b` in your terminal.")
        else:
            st.markdown(f'<div class="status-badge status-success">🟢 Connected (Local)</div>', unsafe_allow_html=True)
            st.caption(f"Active model: **{ollama_handler.model}**")
    else:
        st.markdown(f'<div class="status-badge status-error">🔴 Offline</div>', unsafe_allow_html=True)
        st.error(conn_msg)
        
    st.markdown("---")
    
    # 2. File Uploader
    st.markdown("### 📂 Load Study Materials")
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, DOC)", 
        type=["pdf", "docx", "doc"], 
        help="Upload text files for indexing and learning analysis."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        index_btn = st.button("⚡ Index File", use_container_width=True)
    with col2:
        clear_db_btn = st.button("🗑️ Reset DB", use_container_width=True)
        
    # Handle indexing file
    if index_btn:
        if uploaded_file is not None:
            with st.spinner("Processing file, chunking and generating embeddings..."):
                # Save temp copy
                temp_dir = os.path.join(os.getcwd(), ".temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Parse document and generate chunks (size 1000, overlap 200)
                    chunks = process_document(temp_path)
                    
                    if chunks:
                        store.add_chunks(chunks)
                        st.success(f"Indexed successfully! Generated {len(chunks)} text chunks.")
                    else:
                        st.error("No text could be extracted from the file.")
                except Exception as e:
                    st.error(f"Error during parsing: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.error("Please upload a file first!")
            
    # Handle Database reset
    if clear_db_btn:
        store.clear()
        st.session_state.chat_history = []
        st.session_state.notes = ""
        st.session_state.quiz_questions = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = {}
        st.success("All local documents and search indices cleared!")

    # 3. Document info/stats
    st.markdown("---")
    st.markdown("### 📊 Index Statistics")
    total_chunks = len(store.chunks)
    st.metric(label="Total Indexed Chunks", value=total_chunks)
    
    if total_chunks > 0:
        unique_sources = list(set([c["metadata"]["source"] for c in store.chunks]))
        st.markdown("**Indexed Documents:**")
        for src in unique_sources:
            st.markdown(f"- 📄 `{src}`")
            
# ----------------- MAIN LAYOUT -----------------
st.markdown("<h1 class='title-gradient'>EduMind AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #818CF8; font-weight:400;'>Intelligent Document Brain and Local Study Assistant</h4>", unsafe_allow_html=True)
st.write("")

# Create three navigation tabs for Q&A, study summaries, and interactive quizzes
tab_qa, tab_notes, tab_quiz = st.tabs([
    "💬 Q&A Document Brain", 
    "📝 Auto-Notes Generator", 
    "✍️ Quiz & MCQ Practice"
])

# ------------- TAB 1: Q&A Document Brain -------------
with tab_qa:
    st.markdown("### 💬 Ask Questions to your Document")
    st.markdown("Chat directly with your uploaded documents using local semantic search and llama3.2:1b.")
    
    if len(store.chunks) == 0:
        st.info("💡 To start, upload a document in the sidebar and click **Index File**!")
    else:
        # Display chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "citations" in message and message["citations"]:
                    with st.expander("📚 View References", expanded=False):
                        for cite in message["citations"]:
                            st.markdown(f'<div class="citation-card"><strong>{cite["source"]} (Chunk {cite["chunk_index"]}):</strong><br/><em>{cite["text"]}</em></div>', unsafe_allow_html=True)
                            
        # User input query
        query = st.chat_input("Ask about your study material...")
        if query:
            # 1. Display user query
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # 2. Query vector store
            with st.spinner("Searching document brain..."):
                retrieved_chunks = store.query(query, top_k=3)
                
            # 3. Stream response from LLM
            with st.chat_message("assistant"):
                if not retrieved_chunks:
                    msg = "I'm sorry, but I couldn't find any relevant sections in the indexed document to answer your question."
                    st.write(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg, "citations": []})
                else:
                    sys_p, user_p = get_qa_prompt(retrieved_chunks, query)
                    
                    try:
                        # Stream the token stream
                        response_stream = ollama_handler.generate_chat(sys_p, user_p, stream=True)
                        full_response = st.write_stream(response_stream)
                        
                        # Display citations dynamically in expander
                        citations = [
                            {
                                "source": chunk["metadata"]["source"],
                                "chunk_index": chunk["metadata"]["chunk_index"],
                                "text": chunk["text"]
                            }
                            for chunk in retrieved_chunks
                        ]
                        
                        with st.expander("📚 View References", expanded=False):
                            for cite in citations:
                                st.markdown(f'<div class="citation-card"><strong>{cite["source"]} (Chunk {cite["chunk_index"]}):</strong><br/><em>{cite["text"]}</em></div>', unsafe_allow_html=True)
                        
                        # Add response to session state chat logs
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_response,
                            "citations": citations
                        })
                    except Exception as e:
                        st.error(f"Failed to generate answer: {str(e)}")

# ------------- TAB 2: Auto-Notes Generator -------------
with tab_notes:
    st.markdown("### 📝 Study Notes Generator")
    st.markdown("Compile study guides, summaries, and key concept outlines directly from your uploaded materials.")
    
    if len(store.chunks) == 0:
        st.info("💡 Index a file in the sidebar to generate comprehensive study notes.")
    else:
        # Check if notes are already in session state
        if st.session_state.notes:
            st.markdown("#### Generated Study Guide")
            st.markdown(st.session_state.notes)
            
            # Provide option to download the notes
            st.download_button(
                label="📥 Download Notes (.md)",
                data=st.session_state.notes,
                file_name="edumind_study_notes.md",
                mime="text/markdown"
            )
            
            if st.button("🔄 Regenerate Notes"):
                st.session_state.notes = ""
                st.rerun()
        else:
            st.write("Click below to analyze the entire document context and assemble notes.")
            if st.button("⚡ Generate Structured Study Notes"):
                # Use first few chunks of the document to represent overall context
                # To prevent overloading, we take the top 6 chunks
                total_to_take = min(6, len(store.chunks))
                notes_chunks = store.chunks[:total_to_take]
                
                sys_p, user_p = get_notes_prompt(notes_chunks)
                
                with st.spinner("Analyzing document structure and compiling study notes..."):
                    try:
                        response_stream = ollama_handler.generate_chat(sys_p, user_p, stream=True)
                        full_notes = st.write_stream(response_stream)
                        st.session_state.notes = full_notes
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error compiling study notes: {str(e)}")

# ------------- TAB 3: Quiz & MCQ Practice -------------
with tab_quiz:
    st.markdown("### ✍️ Interactive Quiz Generator")
    st.markdown("Test your comprehension with customized multiple-choice tests generated from your materials.")
    
    if len(store.chunks) == 0:
        st.info("💡 Upload and Index a document in the sidebar to generate interactive practice quizzes.")
    else:
        # Configuration
        num_q = st.slider("Select number of questions", min_value=2, max_value=5, value=3)
        
        generate_quiz_btn = st.button("📝 Generate Practice Quiz")
        
        if generate_quiz_btn:
            # Pick a subset of chunks (representing different portions of the document)
            # To get variety, let's pick up to 5 chunks spaced evenly
            total_ch = len(store.chunks)
            indices = [int(i * total_ch / min(5, total_ch)) for i in range(min(5, total_ch))]
            quiz_chunks = [store.chunks[idx] for idx in indices]
            
            sys_p, user_p = get_quiz_prompt(quiz_chunks, num_questions=num_q)
            
            with st.spinner("Crafting quiz questions..."):
                try:
                    quiz_raw = ollama_handler.generate_chat(sys_p, user_p, stream=False, json_format=True)
                    # Parse output
                    quiz_data = None
                    try:
                        quiz_parsed = json.loads(quiz_raw)
                        if isinstance(quiz_parsed, dict) and "quiz" in quiz_parsed:
                            quiz_data = quiz_parsed["quiz"]
                        elif isinstance(quiz_parsed, list):
                            quiz_data = quiz_parsed
                    except Exception:
                        pass
                        
                    if quiz_data:
                        st.session_state.quiz_questions = quiz_data
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                    else:
                        st.error("Ollama generated invalid quiz structure. Please try generating again.")
                        st.text(f"Raw Output: {quiz_raw}")
                except Exception as e:
                    st.error(f"Failed to generate quiz: {str(e)}")
                    
        # Render the quiz if it exists in session state
        if st.session_state.quiz_questions:
            st.markdown("---")
            st.markdown("#### 📝 Practice Quiz")
            
            for idx, q_item in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Question {idx + 1}: {q_item.get('question', '')}**")
                
                options = q_item.get("options", [])
                
                # Fetch previous answer if saved
                selected_val = st.session_state.quiz_answers.get(idx, None)
                
                # Render radio options
                choice = st.radio(
                    label=f"Options for Question {idx + 1}",
                    options=options,
                    index=options.index(selected_val) if selected_val in options else None,
                    key=f"q_radio_{idx}",
                    label_visibility="collapsed"
                )
                
                # Save choice to session state
                if choice:
                    st.session_state.quiz_answers[idx] = choice
                    
                st.markdown("")
                
            # Submit Button
            if not st.session_state.quiz_submitted:
                if st.button("✔️ Submit Answers"):
                    # Check if all questions are answered
                    if len(st.session_state.quiz_answers) < len(st.session_state.quiz_questions):
                        st.warning("Please answer all questions before submitting!")
                    else:
                        st.session_state.quiz_submitted = True
                        st.rerun()
            else:
                st.markdown("---")
                st.markdown("### 📊 Quiz Results")
                
                score = 0
                for idx, q_item in enumerate(st.session_state.quiz_questions):
                    correct = q_item.get("correct_answer")
                    selected = st.session_state.quiz_answers.get(idx)
                    
                    st.markdown(f"**Question {idx + 1}: {q_item.get('question')}**")
                    st.markdown(f"- Your selection: `{selected}`")
                    st.markdown(f"- Correct answer: `{correct}`")
                    
                    # Exact or partial match check
                    if selected == correct:
                        score += 1
                        st.markdown('<div class="status-badge status-success">✅ Correct</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-badge status-error">❌ Incorrect</div>', unsafe_allow_html=True)
                        
                    st.markdown(f"*Explanation: {q_item.get('explanation')}*")
                    st.markdown("---")
                    
                # Display final grading
                pct = int((score / len(st.session_state.quiz_questions)) * 100)
                st.subheader(f"Total Score: {score} / {len(st.session_state.quiz_questions)} ({pct}%)")
                
                if pct == 100:
                    st.balloons()
                    st.success("🌟 Perfect score! You've mastered this material!")
                elif pct >= 70:
                    st.info("👍 Great job! You understand most of the core concepts.")
                else:
                    st.warning("📖 Good effort! Re-read the notes and try again to improve.")
                    
                if st.button("🔄 Reset Quiz"):
                    st.session_state.quiz_questions = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.rerun()
