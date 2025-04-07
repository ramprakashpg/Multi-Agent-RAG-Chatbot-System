# Wikipedia wrapper
import requests
from bs4 import BeautifulSoup
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

wiki = WikipediaAPIWrapper()
llm = Ollama(model="llama3.2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query = "Artificial Intelligence"
wiki_result = wiki.run(query)

memory_agent_2 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Fetch the webpage

def extract_article_content(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    content_section = soup.find("div")
    if not content_section:
        return {"content": "No content found."}

    paragraphs = content_section.find_all("p")
    content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text())

    return content


latest_news = extract_article_content("https://www.britannica.com/technology/artificial-intelligence/Reasoning")

# Fetch from a PDF file
loader = PyPDFLoader("assests/ai_book_1.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Creating ChromaDB directory
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data = text_splitter.create_documents([wiki_result, latest_news])

ai_store = Chroma(persist_directory="./chroma_ai", embedding_function=embedding_model)

ai_store.add_documents(data)
ai_store.add_documents(docs)
ai_retriever = ai_store.as_retriever()
ai_chain = ConversationalRetrievalChain.from_llm(llm, ai_retriever, memory=memory_agent_2,
                                                 output_key="answer")


def ai_qa(prompt):
    return ai_chain.run({"question": prompt})
