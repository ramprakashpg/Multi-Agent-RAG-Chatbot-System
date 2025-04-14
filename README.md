# 🧠 Multi-Agent RAG Chatbot System

A college project exploring the power of **multi-agent systems** and **Retrieval-Augmented Generation (RAG)** using cutting-edge tools like **LangChain**, **Ollama**, and **ChromaDB**. This chatbot leverages multiple specialized agents to collaboratively process user queries, fetch context from PDFs/web sources, and generate accurate, context-aware responses.

---

## 🧑‍💻 Meet the Agents

This system uses a **multi-agent design** with three specialized agents, each focused on a unique domain of knowledge:

### 1. 🧠 General Agent
Handles open-domain, casual, or general-purpose questions. This is the default fallback agent if no specific domain is detected in the query.

### 2. 🤖 AI Specialist Agent
Focused on answering questions related to Artificial Intelligence—LLMs, machine learning concepts, tools like LangChain, model training, etc.

### 3. 🎓 Concordia Helpdesk
Provides information about the **Computer Science Department at Concordia University**, such as:
- Course offerings
- Faculty details
- Admission requirements
- Labs and facilities

Each query is **routed dynamically** based on its content. The **controller agent** analyzes the question and delegates it to the most relevant specialist, ensuring high-quality, domain-specific answers.


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
    │   ├── chat-agents/            # LangChain agent logic
    |       ├── ai_llmservice.py  
    |       └── concordia_llm_service.py 
    |       └── general_assistant.py
    |   ├── utils/
    │   └── GeneralAgentSARSA.py  # Reinforcement Learning Model
    ├── database/                # Chroma db which contains embeddings
    |   └── chroma_general/
    |   └── chroma_ai/
    |   └── chroma_concordia/
    ├── frontend/
    │   └── chat_app.py           # Streamlit UI
    ├── assests/
    │   └── ai_book_1.pdf         # PDFs or scraped data
    ├── requirements.txt
    ├── app_server.py             # FastAPI Server
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

## 📸 Screenshots

### Home Page with access to multiple agents:
<img width="1335" alt="image" src="https://github.com/user-attachments/assets/e53c95da-3768-4859-9ac9-6a30448b6b8a" />

The chatbot uses **ConversationalMemoryBuffer** to understand the context from the previous chat history.
<img width="1335" alt="image" src="https://github.com/user-attachments/assets/cc3f9fa5-c755-4d7b-8205-4084b0f3f367" />











