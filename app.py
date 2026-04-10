"""
app.py — Streamlit UI for the RAG PDF Chatbot
===============================================
This is the front-end of our application.
It handles:
- PDF file upload (multiple files)
- Processing PDFs (building the vector store)
- Chat interface with styled message bubbles
- Source citation display
- Session state management (keeps data between reruns)

🧠 How Streamlit works:
   Every time you interact with a widget (button, text box, file uploader),
   Streamlit re-runs the ENTIRE script from top to bottom.
   We use st.session_state (a Python dict) to "remember" things between reruns.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Import our backend functions
from rag_backend import build_vectorstore, get_answer

# ================================================================
# Load environment variables (.env file)
# ================================================================
load_dotenv()

# ================================================================
# Page Configuration
# ================================================================
st.set_page_config(
    page_title=" PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# Custom CSS — Styling for chat bubbles and overall look
# ================================================================
st.markdown("""
<style>
    /* ---- Import modern font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---- Global styles ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- App background ---- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ---- Main container ---- */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* ---- Header ---- */
    .chat-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    .chat-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .chat-header p {
        color: rgba(255,255,255,0.6);
        margin: 0.4rem 0 0;
        font-size: 1rem;
    }

    /* ---- Chat message bubbles ---- */
    .message-container {
        display: flex;
        margin: 0.8rem 0;
        align-items: flex-start;
        gap: 10px;
    }

    /* User message — right aligned */
    .user-message-container {
        flex-direction: row-reverse;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 4px 18px 18px;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    /* AI message — left aligned */
    .ai-bubble {
        background: rgba(255,255,255,0.1);
        color: #f0f0f0;
        padding: 12px 18px;
        border-radius: 4px 18px 18px 18px;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.6;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(5px);
    }

    /* Avatars */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .user-avatar { background: linear-gradient(135deg, #667eea, #764ba2); }
    .ai-avatar   { background: linear-gradient(135deg, #f093fb, #f5576c); }

    /* ---- Status badge ---- */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .status-ready  { background: rgba(46,213,115,0.2); color: #2ed573; border: 1px solid #2ed573; }
    .status-upload { background: rgba(255,165,2,0.2);  color: #ffa502; border: 1px solid #ffa502; }

    /* ---- Sidebar styling ---- */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.8);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #ffffff;
        font-size: 1.1rem;
    }

    /* ---- Input box ---- */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.4); }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }

    /* ---- Source citations expander ---- */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.85rem !important;
    }
    .source-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.75);
        line-height: 1.5;
    }
    .source-label {
        color: #667eea;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 4px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)


# ================================================================
# Session State Initialisation
# ================================================================
# st.session_state persists between re-runs (unlike regular variables).
# Think of it like a notepad Streamlit keeps for the whole session.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of (user_msg, ai_msg) tuples

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None         # FAISS vector store

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False     # flag: have we processed PDFs?

if "processed_pdf_names" not in st.session_state:
    st.session_state.processed_pdf_names = []   # track file names


# ================================================================
# Sidebar — PDF Upload & Controls
# ================================================================
with st.sidebar:
    st.markdown("## 📁 Upload Documents")
    st.markdown("---")

    # Multiple file uploader — accepts one or more PDFs
    uploaded_files = st.file_uploader(
        label="Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once."
    )

    st.markdown("---")

    # Process button — triggers building the vector store
    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one PDF first.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error(
                "❌ **API Key Missing!**\n\n"
                "Create a `.env` file in the project folder with:\n"
                "```\nGOOGLE_API_KEY=your_key_here\n```\n"
                "Get your free key at: https://aistudio.google.com/app/apikey"
            )
        else:
            # Save uploaded files to a temp directory so PyPDFLoader can read them.
            # (Streamlit gives us file bytes, but PyPDFLoader needs a file path.)
            with st.spinner("🔄 Processing PDFs... this may take a moment"):
                try:
                    temp_dir = tempfile.mkdtemp()
                    pdf_paths = []

                    for uploaded_file in uploaded_files:
                        # Write each PDF to a temporary file
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(temp_path)

                    # Build the FAISS vector store from all PDFs
                    st.session_state.vectorstore = build_vectorstore(pdf_paths)
                    st.session_state.pdfs_processed = True
                    st.session_state.processed_pdf_names = [f.name for f in uploaded_files]
                    # Clear old chat when new PDFs are loaded
                    st.session_state.chat_history = []

                    st.success(f"✅ Processed {len(uploaded_files)} PDF(s) successfully!")

                except ValueError as e:
                    st.error(f"⚠️ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}\n\nCheck your API key and try again.")

    st.markdown("---")

    # Show which PDFs are currently loaded
    if st.session_state.pdfs_processed and st.session_state.processed_pdf_names:
        st.markdown("**📌 Loaded Documents:**")
        for name in st.session_state.processed_pdf_names:
            st.markdown(f"-  `{name}`")
    else:
        st.markdown(
            '<div style="color:rgba(255,255,255,0.4); font-size:0.85rem;">'
            'No documents loaded yet.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Reset / clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Helpful info
    st.markdown("---")
    st.markdown(
        '<div style="color:rgba(255,255,255,0.4); font-size:0.78rem; line-height:1.6;">'
        '💡 <b>Tips:</b><br>'
        '• Upload PDFs → Click Process<br>'
        '• Ask specific questions<br>'
        '• Follow-up questions work too!<br>'
        '• Citations show the source text'
        '</div>',
        unsafe_allow_html=True
    )


# ================================================================
# Main Area — Header
# ================================================================
st.markdown("""
<div class="chat-header">
    <h1>🤖 PDF Chatbot</h1>
    <p>Upload a PDF, ask anything — powered by RAG + Gemini</p>
</div>
""", unsafe_allow_html=True)

# Status badge — shows if PDFs are ready or not
if st.session_state.pdfs_processed:
    n = len(st.session_state.processed_pdf_names)
    st.markdown(
        f'<span class="status-badge status-ready">● Ready — {n} document(s) loaded</span>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="status-badge status-upload">● Upload PDFs in the sidebar to begin</span>',
        unsafe_allow_html=True
    )


# ================================================================
# Chat History Display
# ================================================================
# Render all previous messages as styled bubbles

if st.session_state.chat_history:
    for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        # -- User bubble (right side) --
        st.markdown(f"""
        <div class="message-container user-message-container">
            <div class="avatar user-avatar">👤</div>
            <div class="user-bubble">{user_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        # -- AI bubble (left side) --
        st.markdown(f"""
        <div class="message-container">
            <div class="avatar ai-avatar">🤖</div>
            <div class="ai-bubble">{ai_msg}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state — shown before any messages
    if st.session_state.pdfs_processed:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color:rgba(255,255,255,0.3);">
            <div style="font-size:3rem;">💬</div>
            <div style="margin-top:0.5rem;">Ask your first question below!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color:rgba(255,255,255,0.3);">
            <div style="font-size:3rem;"></div>
            <div style="margin-top:0.5rem;">Upload a PDF in the sidebar, then ask anything!</div>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# Chat Input + Source Citations
# ================================================================
st.markdown("---")

# Use a form so pressing Enter submits the message
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_question = st.text_input(
            label="Your question",
            placeholder="Ask something about your PDF...",
            label_visibility="collapsed"
        )
    with col_btn:
        submit_btn = st.form_submit_button("Send 🚀", use_container_width=True)


# Handle submission
if submit_btn:
    # --- Validation guards ---
    if not user_question.strip():
        st.warning("⚠️ Please type a question before sending.")

    elif not st.session_state.pdfs_processed or st.session_state.vectorstore is None:
        st.info("Please upload and process at least one PDF first (use the sidebar).")

    else:
        # Show a spinner while Gemini is thinking
        with st.spinner("🤔 Thinking..."):
            try:
                answer, source_docs, updated_history = get_answer(
                    vectorstore=st.session_state.vectorstore,
                    question=user_question,
                    chat_history=st.session_state.chat_history
                )

                # Save updated history to session state
                st.session_state.chat_history = updated_history

                # Display the latest exchange immediately
                st.markdown(f"""
                <div class="message-container user-message-container">
                    <div class="avatar user-avatar">👤</div>
                    <div class="user-bubble">{user_question}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="message-container">
                    <div class="avatar ai-avatar">🤖</div>
                    <div class="ai-bubble">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                # -- Source Citations --
                # Show the actual text chunks used to generate the answer
                if source_docs:
                    with st.expander(f"📚 View {len(source_docs)} Source(s) Used", expanded=False):
                        seen_content = set()   # avoid showing duplicate chunks
                        for j, doc in enumerate(source_docs):
                            content_snippet = doc.page_content[:150]
                            if content_snippet in seen_content:
                                continue
                            seen_content.add(content_snippet)

                            # Extract metadata (page number, source file)
                            page_num = doc.metadata.get("page", "?")
                            source_file = os.path.basename(
                                doc.metadata.get("source", "Unknown")
                            )

                            st.markdown(
                                f'<div class="source-card">'
                                f'<div class="source-label">{source_file} — Page {page_num + 1}</div>'
                                f'{doc.page_content[:300]}{"..." if len(doc.page_content) > 300 else ""}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                # Rerun to refresh the full chat display
                st.rerun()

            except Exception as e:
                error_msg = str(e)
                if "API_KEY" in error_msg.upper() or "api key" in error_msg.lower():
                    st.error(
                        "❌ **API Key Error**: Your Gemini API key is invalid or missing.\n\n"
                        "Check your `.env` file and make sure the key is correct."
                    )
                elif "quota" in error_msg.lower() or "429" in error_msg:
                    st.error(
                        "⏳ **Rate Limit**: You've hit the Gemini API rate limit. "
                        "Wait a moment and try again."
                    )
                else:
                    st.error(f"❌ An error occurred: {error_msg}")
