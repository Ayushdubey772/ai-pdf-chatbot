# 🤖 Mini RAG PDF Chatbot

A fully-featured AI chatbot powered by **Retrieval-Augmented Generation (RAG)** that lets you upload PDF documents and ask questions about them in natural language.

![Tech Stack](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green) ![Gemini](https://img.shields.io/badge/Google%20Gemini-API-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)

---

## 🧠 What is RAG?

> **Analogy:** Imagine you give a brilliant friend a book to read. When you ask them a question, they flip through the book, find the most relevant pages, and then answer you in their own words — citing where they found it.

That's RAG:
1. **Retrieval** — Find the most relevant chunks of your PDF
2. **Augmented** — Feed those chunks as context to an LLM
3. **Generation** — The LLM generates a precise, grounded answer

This prevents the LLM from hallucinating (making things up) because it's anchored to your actual document.

---

## ✨ Features

| Feature | Status |
|---|---|
| Upload multiple PDFs | ✅ |
| Smart text chunking | ✅ |
| Semantic embeddings (Google) | ✅ |
| FAISS vector search | ✅ |
| Gemini-powered answers | ✅ |
| Conversational memory | ✅ |
| Source citations with page numbers | ✅ |
| Beautiful dark chat UI | ✅ |
| Error handling & user guidance | ✅ |

---

## 📁 Project Structure

```
pdf_chatbot/
├── app.py              → Streamlit web interface
├── rag_backend.py      → Core RAG logic (AI pipeline)
├── requirements.txt    → Python dependencies
├── .env.example        → API key template
├── .gitignore          → Excludes secrets & temp files
└── README.md           → This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- A free Google Gemini API key

### Step 1: Get your API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key

### Step 2: Clone / Download the project
```bash
# If using Git:
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot

# Or just download and unzip, then open a terminal in the folder
```

### Step 3: Create a virtual environment
```bash
# Create venv
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

> ⚠️ **Common Mistake:** Always activate your virtual environment before installing or running anything. You'll see `(venv)` at the start of your terminal prompt when it's active.

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Set up your API key
```bash
# Copy the template
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux

# Open .env and replace the placeholder with your real key:
# GOOGLE_API_KEY=AIza...your_actual_key_here
```

### Step 6: Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501` 🎉

---

## 🎯 How to Use

1. **Upload PDFs** — Use the sidebar to upload one or more PDF files
2. **Process** — Click the **"⚡ Process PDFs"** button and wait for the progress to complete
3. **Ask questions** — Type any question about your documents in the chat box
4. **View citations** — Expand the **"📚 View Sources"** section below each answer to see exactly which page the answer came from
5. **Follow-up questions** — The chatbot remembers context, so you can ask follow-up questions naturally

---

## 🏗️ Architecture

```
User uploads PDF(s)
        ↓
  PyPDFLoader (page-by-page extraction)
        ↓
  RecursiveCharacterTextSplitter
  (chunk_size=1000, overlap=200)
        ↓
  GoogleGenerativeAIEmbeddings
  (text-embedding-004)
        ↓
  FAISS Vector Store
  (stored in memory)
        ↓
  User asks a question
        ↓
  Similarity Search → Top 5 chunks
        ↓
  ChatGoogleGenerativeAI (gemini-1.5-flash)
  + ConversationalRetrievalChain + Memory
        ↓
  Answer + Source Citations
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **LangChain** | RAG orchestration framework |
| **FAISS** | Fast vector similarity search |
| **Google Gemini API** | LLM (answers) + Embeddings |
| **Streamlit** | Interactive web UI |
| **python-dotenv** | Secure API key management |
| **PyPDF** | PDF text extraction |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY not found` | Make sure `.env` exists with your key (not `.env.example`) |
| `No module named 'langchain'` | Run `pip install -r requirements.txt` with venv activated |
| Empty answer / "not in documents" | Try rephrasing the question or use simpler terms |
| Rate limit error (429) | Wait 60 seconds and retry — you're on the free tier |
| PDF text not extracted | Ensure your PDF is text-based (not a scanned image PDF) |

---

##  License

MIT License — feel free to use this for learning, your portfolio, or as a base for your own projects.

---

## 🙋 Built With Love

This project demonstrates the power of **Retrieval-Augmented Generation** — a technique used by production AI systems like ChatGPT Enterprise and Google's enterprise search. Perfect for your portfolio!

> 💡 **What to say in interviews:** "I built a RAG chatbot that uses FAISS for semantic search and Google's Gemini API as the LLM backbone. The system chunks documents, embeds them into vectors, retrieves the most similar chunks to the query, and grounds the LLM's response in the actual document content — reducing hallucinations."
