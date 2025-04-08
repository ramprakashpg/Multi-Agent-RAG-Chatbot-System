from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM

from backend.utils.GeneralAgentSARSA import GeneralAgentSARSA

llm = OllamaLLM(model="llama3.2")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory="database/chroma_general", embedding_function=embedding_model)
retriever = vector_store.as_retriever()

# Create LangChain's ConversationRetrievalChain
llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# List of possible actions for the agent
actions_general = [
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

# Create the agent with SARSA, LangChain, and ChromaDB
agent = GeneralAgentSARSA(actions_general, llm_chain, q_table_path="backend/logs/SARSA_q_table_general.npy",
                          log_file="backend/logs/queries_responses_general.xlsx")


def get_response(user_input):
    # Process the user input and get a response
    response = agent.process_input(user_input)
    return response


# For handling explicit feedback (thumbs-up and thumbs-down)
def update_feedback(feedback_type):
    reward = agent.handle_feedback(feedback_type.lower())
