import requests
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory="./chroma_general", embedding_function=embedding_model)
retriever = vector_store.as_retriever()

qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)


def general_qa(prompt: str):
    return qa_chain.invoke({"question": prompt})


# Wikipedia wrapper
wiki = WikipediaAPIWrapper()
query = "Artificial Intelligence"
wiki_result = wiki.run(query)

memory_agent_2 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Fetch the webpage
url = "https://ai.googleblog.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
latest_news = soup.find("h2").text

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


response = ai_qa("What is AI?")
print(response)
print(ai_qa("Explain more about it?"))
