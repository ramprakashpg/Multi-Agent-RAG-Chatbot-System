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
    loader = PyPDFLoader(file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    return docs


def extract_article_content(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    content_section = soup.find("div")
    if not content_section:
        return {"content": "No content found."}

    paragraphs = content_section.find_all("p")
    content = "\n ".join(p.get_text(strip=True) for p in paragraphs if p.get_text())
    return content


article_1 = extract_article_content("https://www.concordia.ca/academics/undergraduate/computer-science.html")
article_2 = extract_article_content(
    "https://www.concordia.ca/academics/undergraduate/calendar/current/section-71-gina-cody-school-of-engineering-and-computer-science/section-71-70-department-of-computer-science-and-software-engineering/section-71-70-2-degree-requirements-bcompsc-.html")
pdf_1 = load_pdf("assests/concordia_1.pdf")
pdf_2 = load_pdf("assests/degree_req.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data = text_splitter.create_documents([article_1, article_2])
concordia_llm = Chroma(persist_directory="database/chroma_concordia", embedding_function=embedding_model)
# concordia_llm.add_documents(data)
concordia_llm.add_documents(pdf_1)
concordia_llm.add_documents(pdf_2)
concordia_llm_retriever = concordia_llm.as_retriever()
retrieval_chain = ConversationalRetrievalChain.from_llm(llm, concordia_llm_retriever, memory=memory,
                                                        output_key="answer")

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
    return concordia_agent.process_input(prompt)


def update_feedback(feedback_type):
    reward = concordia_agent.handle_feedback(feedback_type.lower())
