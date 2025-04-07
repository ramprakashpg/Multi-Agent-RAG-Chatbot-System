from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
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
    return qa_chain.run({"question": prompt})






