"""
rag_backend.py — The Brain of the PDF Chatbot
================================================
Rewritten for LangChain 1.x using LCEL (LangChain Expression Language).

Pipeline:
  PDF files → load pages → split into chunks → embed → FAISS
  On each query: FAISS similarity search → Gemini LLM → answer + sources

🧠 Analogy: Think of this as a smart librarian.
   - PDFs = books in the library
   - Chunks = individual paragraphs (flash cards)
   - Embeddings = unique "fingerprint" of each chunk
   - FAISS = card catalogue for finding similar paragraphs
   - Gemini = librarian who reads the best matches and writes your answer
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import tempfile
from typing import List, Tuple, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- PDF loading (reads PDF page by page into Document objects) ---
from langchain_community.document_loaders import PyPDFLoader

# --- Text splitting (breaks long docs into overlapping chunks) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Google embeddings (text → vector of numbers capturing meaning) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Gemini chat LLM ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- FAISS vector store ---
from langchain_community.vectorstores import FAISS

# --- LCEL core components ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ================================================================
# STEP 1 & 2: Load PDFs and split into chunks
# ================================================================

def load_and_split_pdfs(pdf_file_paths: List[str]) -> List[Any]:
    """
    Load one or more PDFs and split them into text chunks.

    🧠 Like cutting a long book into flash cards.
       Each card = one chunk. Easier to search than 100 pages.

    Args:
        pdf_file_paths: List of absolute paths to PDF files.
    Returns:
        List of LangChain Document objects (chunks + metadata).
    """
    all_documents = []

    for pdf_path in pdf_file_paths:
        print(f"Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"   ✅ {len(docs)} pages loaded")

    # chunk_size=1000  → each chunk ≤ 1000 characters
    # chunk_overlap=200 → 200-char overlap between adjacent chunks
    # (overlap preserves context at chunk boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(all_documents)
    print(f"✂️  {len(chunks)} chunks from {len(all_documents)} pages")
    return chunks


# ================================================================
# STEP 3, 4 & 5: Embeddings + FAISS vector store
# ================================================================

def build_vectorstore(pdf_file_paths: List[str]) -> FAISS:
    """
    Build a FAISS vector database from uploaded PDFs.

    Steps inside:
    1. Load & split PDFs → chunks
    2. Embed each chunk (text → 768-dim vector via Gemini embedding model)
    3. Store vectors in FAISS for fast similarity search

    Args:
        pdf_file_paths: Paths to saved PDF files.
    Returns:
        FAISS vector store ready for retrieval.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GOOGLE_API_KEY not found! "
            "Create a .env file with GOOGLE_API_KEY=your_key_here. "
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    chunks = load_and_split_pdfs(pdf_file_paths)
    if not chunks:
        raise ValueError("❌ No text could be extracted from the uploaded PDFs.")

    print("\n🔢 Generating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key
    )

    # FAISS.from_documents: embeds every chunk + stores in FAISS index
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print("✅ Vector database ready!")
    return vectorstore


# ================================================================
# STEP 6 & 7: Gemini LLM + LCEL Retrieval Chain
# ================================================================

def get_answer(
    vectorstore: FAISS,
    question: str,
    chat_history: List[Tuple[str, str]]
) -> Tuple[str, List[Any], List[Tuple[str, str]]]:
    """
    Ask a question and get a grounded answer with source documents.

    Uses a two-step LCEL chain:
    1. Condense question + history → standalone question
       (so follow-ups like "What about that?" become self-contained)
    2. Retrieve top chunks → feed to Gemini → generate answer

    Args:
        vectorstore:  Loaded FAISS store.
        question:     User's question string.
        chat_history: [(human_msg, ai_msg), ...] so far.

    Returns:
        (answer, source_docs, updated_chat_history)
    """
    if not question.strip():
        return "Please enter a question.", [], chat_history

    api_key = os.getenv("GOOGLE_API_KEY")

    # --- Gemini LLM ---
    # temperature=0.3: mostly factual, slightly flexible
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    # --- Retriever: MMR with k=5 avoids repetitive chunks ---
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    # ── Chain 1: Condense history + new question into standalone question ──
    # This makes follow-up questions like "And what about X?" work correctly.
    condense_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human",
         "Given the conversation above, rephrase the follow-up question as a "
         "standalone question that can be understood without any prior context. "
         "If no rephrasing is needed, return the question as-is.")
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    # ── Chain 2: Answer question using retrieved context ──
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful AI assistant that answers questions based on the provided PDF documents.

Use ONLY the following context to answer. If the answer is not in the context, say:
"I couldn't find that information in the uploaded documents."

Be concise, accurate, and cite the document section when possible.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Helper: format retrieved Document objects into a single string
    def format_docs(docs: List[Any]) -> str:
        return "\n\n".join(
            f"[Page {d.metadata.get('page', '?') + 1} | "
            f"{os.path.basename(d.metadata.get('source', 'doc'))}]\n"
            f"{d.page_content}"
            for d in docs
        )

    # ── Convert chat_history tuples → LangChain message objects ──
    lc_history = []
    for human_msg, ai_msg in chat_history:
        lc_history.append(HumanMessage(content=human_msg))
        lc_history.append(AIMessage(content=ai_msg))

    # ── Step 1: Build standalone question ──
    # Skip condensing if there's no history (first message)
    if chat_history:
        standalone_question = condense_chain.invoke({
            "chat_history": lc_history,
            "input": question
        })
    else:
        standalone_question = question

    # ── Step 2: Retrieve + Answer ──
    source_docs = retriever.invoke(standalone_question)
    context_text = format_docs(source_docs)

    answer_chain = qa_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "chat_history": lc_history,
        "input": standalone_question,
        "context": context_text
    })

    updated_history = chat_history + [(question, answer)]
    return answer, source_docs, updated_history
