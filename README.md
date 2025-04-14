# 🧠 Multi-Agent RAG Chatbot System

A college project exploring the power of **multi-agent systems** and **Retrieval-Augmented Generation (RAG)** using cutting-edge tools like **LangChain**, **Ollama**, and **ChromaDB**. This chatbot leverages multiple specialized agents to collaboratively process user queries, fetch context from PDFs/web sources, and generate accurate, context-aware responses.

---

## 🚀 Features

- 🤖 **Multi-Agent Architecture**: Agents with specialized roles (Retriever, Summarizer, Answerer, etc.)
- 📚 **RAG Pipeline**: Combines real-time document/web data with LLM responses
- 📄 **PDF + Web Scraping Support**: Extracts context from uploaded files or URLs
- 🔍 **Vector Search**: Uses ChromaDB for semantic document retrieval
- 🧠 **Local LLMs**: Runs large language models locally using Ollama
- 🌐 **FastAPI Backend**: Lightweight and scalable RESTful API
- 🎨 **Streamlit Frontend**: Simple, interactive UI for users to chat with the system

---

## 🛠️ Tech Stack

| Layer         | Tool/Framework       | Description                                      |
|--------------|----------------------|--------------------------------------------------|
| LLM Orchestration | LangChain             | Multi-agent coordination + RAG pipeline         |
| LLM Runtime   | Ollama                | Run open-source LLMs like LLaMA locally         |
| Memory Store  | ChromaDB              | Vector database for storing and retrieving docs |
| Backend       | FastAPI               | Serves the API endpoints                        |
| Frontend      | Streamlit             | Provides a user-friendly chat interface         |
| Data Sources  | Web Scraping + PDF    | Collects external data for grounding responses  |

---

## 📂 Project Structure

multi-agent-chatbot/
├── backend/
│   ├── main.py               # FastAPI server
│   └── agents/               # LangChain agent logic
├── frontend/
│   └── app.py                # Streamlit UI
├── data/
│   └── uploads/              # PDFs or scraped data
├── utils/
│   └── scraping.py           # Web scraper
│   └── pdf_loader.py         # PDF parsing and processing
├── requirements.txt
└── README.md


---

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-agent-chatbot.git
cd multi-agent-rag-chatbot-system
```

### 2. Setting up Virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
### 3. Install Requirements
```bash
pip install requirement.txt
```
### 4. Start Ollama (LLM Server) and Run the backend

```bash
ollama run llama3
cd backend
uvicorn main:app --reload
```
## 🧠 How It Works

1. The user enters a query in the **Streamlit** frontend.
2. The query is sent to the **FastAPI** backend.
3. A **controller agent** built using **LangChain** orchestrates the process:
    - 🧾 **Retriever Agent**: Queries **ChromaDB** for relevant information from preprocessed PDFs and web pages.
    - 🌐 **Scraper/Loader Agent**: If needed, scrapes content from provided URLs or loads and parses uploaded PDFs.
    - 🧠 **Answer Generator Agent**: Uses an **LLM via Ollama** to craft a final, context-aware response based on retrieved data.
4. The response is returned via the API and displayed in the Streamlit chat interface.







