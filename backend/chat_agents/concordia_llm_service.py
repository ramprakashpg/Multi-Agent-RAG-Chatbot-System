"""
concordia_chatbot.py

This module sets up a SARSA-based conversational agent tailored to answer queries about
Concordia University's Computer Science undergraduate programs. It leverages LangChain for
retrieval-augmented generation, document embeddings via HuggingFace, and reinforcement learning
to refine conversational strategies over time.

Key features:
- Loads and splits both web and PDF content.
- Stores document embeddings in a Chroma vector store.
- Integrates with a custom SARSA reinforcement learning agent.
- Tracks user feedback and logs interactions to Excel for analysis.

Author: Sasikiran Sivakumar
Date: April 8, 2025
"""
import requests
from bs4 import BeautifulSoup
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.GeneralAgentSARSA import GeneralAgentSARSA

llm = OllamaLLM(model="llama3.2")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Concordia Web page
def load_pdf(file):
    """
        Loads a PDF file, splits it into chunks for vector storage.

        Args:
            file (str): Path to the PDF file.

        Returns:
            List[Document]: A list of split LangChain Document chunks.
        """
    loader = PyPDFLoader(file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    return docs


def extract_article_content(url):
    """
       Extracts textual content from a webpage, focusing on <p> tags within the first <div>.

       Args:
           url (str): URL of the article/page.

       Returns:
           str: Cleaned text content extracted from the page.
       """
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    content_section = soup.find("div")
    if not content_section:
        return {"content": "No content found."}

    paragraphs = content_section.find_all("p")
    content = "\n ".join(p.get_text(strip=True) for p in paragraphs if p.get_text())
    return content


# Initialize core components
article_1 = extract_article_content("https://www.concordia.ca/academics/undergraduate/computer-science.html")
article_2 = extract_article_content(
    "https://www.concordia.ca/academics/undergraduate/calendar/current/section-71-gina-cody-school-of-engineering-and-computer-science/section-71-70-department-of-computer-science-and-software-engineering/section-71-70-2-degree-requirements-bcompsc-.html")
pdf_1 = load_pdf("assests/concordia_1.pdf")
pdf_2 = load_pdf("assests/degree_req.pdf")

# Prepare vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data = text_splitter.create_documents([article_1, article_2])
concordia_llm = Chroma(persist_directory="database/chroma_concordia", embedding_function=embedding_model)

# Add PDFs (text data is skipped for now to avoid redundancy)
concordia_llm.add_documents(pdf_1)
concordia_llm.add_documents(pdf_2)
concordia_llm_retriever = concordia_llm.as_retriever()

# Setup LangChain Conversational Retrieval chain
retrieval_chain = ConversationalRetrievalChain.from_llm(llm, concordia_llm_retriever, memory=memory,
                                                        output_key="answer")

# Define a set of diverse actions for the RL agent to choose from
actions_concordia = [
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
concordia_agent = GeneralAgentSARSA(actions=actions_concordia, llm_chain=retrieval_chain,
                                    q_table_path="backend/logs/SARSA_q_table_concordia.npy",
                                    log_file="backend/logs/queries_responses_concordia.xlsx")


def generate_response(prompt):
    """
        Generates a response from the Concordia agent based on user input.

        Args:
            prompt (str): User query.

        Returns:
            str: LLM-generated response.
        """
    return concordia_agent.process_input(prompt)


def update_feedback(feedback_type):
    """
        Generates a response from the Concordia agent based on user input.

        Args:
            prompt (str): User query.

        Returns:
            str: LLM-generated response.
        """
    reward = concordia_agent.handle_feedback(feedback_type.lower())
