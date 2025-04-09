"""
AI Specialist Module (Wikipedia & Document-Based QA System)

This module sets up an AI-powered question-answering system that combines information retrieval
from Wikipedia, PDF documents, and news articles. It integrates a conversational retrieval chain
using LangChain's LLM, memory, and vector store components.

Key Components:
- WikipediaAPIWrapper for fetching encyclopedia-style info.
- PyPDFLoader and BeautifulSoup for document and web scraping.
- Chroma vector store for semantic search across sources.
- ConversationalRetrievalChain for context-aware conversations.
- A SARSA-based reinforcement learning agent (GeneralAgentSARSA) to model adaptive conversational behavior.

Used by the AI Specialist chatbot in the Streamlit frontend.
Author: Saran Kirthic Sivakumar
Date: April 8, 2025
"""
import requests
from bs4 import BeautifulSoup
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.GeneralAgentSARSA import GeneralAgentSARSA

wiki = WikipediaAPIWrapper()  #: LangChain wrapper to query Wikipedia
llm = OllamaLLM(model="llama3.2")  #: Local LLM instance from Ollama for generation
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query = "Artificial Intelligence"
wiki_result = wiki.run(query)  #: Resulting Wikipedia summary for the query

memory_agent_2 = ConversationBufferMemory(memory_key="chat_history",
                                          return_messages=True)  #: Memory buffer to enable conversational history in retrieval


def extract_article_content(url):
    """
    Scrapes and extracts main text content from a given article URL.

    Args:
        url (str): The full URL of the article to extract content from.

    Returns:
        str: Concatenated paragraph text from the article or a fallback message.
    """
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    content_section = soup.find("div")
    if not content_section:
        return {"content": "No content found."}

    paragraphs = content_section.find_all("p")
    content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text())

    return content


latest_news = extract_article_content(
    "https://www.britannica.com/technology/artificial-intelligence/Reasoning")  #: Scraped content from a Britannica article on AI reasoning

# Fetch from a PDF file
loader = PyPDFLoader("assests/ai_book_1.pdf")
documents = loader.load()  #: Parsed documents from PDF

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)  #: Chunks from PDF documents

# Creating ChromaDB directory
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data = text_splitter.create_documents([wiki_result, latest_news])  #: Chunks from Wikipedia + web content

ai_store = Chroma(persist_directory="database/chroma_ai",
                  embedding_function=embedding_model)  #: Vector store initialized with persistent directory and embedding model

ai_store.add_documents(data)
ai_store.add_documents(docs)
ai_retriever = ai_store.as_retriever()
ai_chain = ConversationalRetrievalChain.from_llm(llm, ai_retriever, memory=memory_agent_2,
                                                 output_key="answer")
actions_ai = [
    "Engage in Casual Conversation",
    "Retrieve Information",
    "Acknowledgment",
    "Small Talk",
    "Compliment/Encouragement",
    "Provide Advice",
    "Ask Clarifying Questions",
    "Show Empathy",
    "Offer Recommendations",
    "Confirm Intentions or Actions",
    "Show Gratitude",
    "Explain Concepts or Ideas",
    "Provide Support",
    "Share Insights or Information",
    "Offer a Summary"
]

ai_agent = GeneralAgentSARSA(actions=actions_ai, llm_chain=ai_chain, q_table_path="backend/logs/SARSA_q_table_ai.npy",
                             log_file="backend/logs/queries_responses_ai.xlsx")


def ai_qa(prompt):
    """
    Processes user input through the AI agent's conversational pipeline and returns a response.

    Args:
        prompt (str): The user's query or message.

    Returns:
        str: The AI agent's generated response based on retrieval and policy.
    """
    return ai_agent.process_input(prompt)


def update_feedback(feedback_type):
    """
        Updates the SARSA-based agent with user feedback to adjust its learning policy.

        Args:
            feedback_type (str): The feedback label ("positive" or "negative").

        Returns:
            None
        """
    reward = ai_agent.handle_feedback(feedback_type.lower())
