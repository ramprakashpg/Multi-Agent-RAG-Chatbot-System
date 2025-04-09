"""
general_agent_module.py

This module sets up a General Assistant chatbot agent using a combination of:
- LangChain's ConversationalRetrievalChain
- HuggingFace sentence embeddings
- ChromaDB for document retrieval
- Ollama LLM for generating responses
- SARSA (a reinforcement learning algorithm) for feedback-driven learning

The agent is capable of retrieving relevant information, engaging in conversation, and adapting behavior
based on positive or negative user feedback.

Author: Ramprakash Periyagaram Ganesan
Date: April 8
"""
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM

from backend.utils.GeneralAgentSARSA import GeneralAgentSARSA

llm = OllamaLLM(model="llama3.2")
"""OllamaLLM: Local language model used to generate responses."""

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
"""ConversationBufferMemory: Stores chat history to enable context-aware conversations."""

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
"""HuggingFaceEmbeddings: Converts documents into vector form for retrieval."""

vector_store = Chroma(persist_directory="database/chroma_general", embedding_function=embedding_model)
"""Chroma: Persistent vector database used to store and retrieve embedded documents."""

retriever = vector_store.as_retriever()
"""Retriever: Interface to search relevant documents from the vector store."""

llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)
"""ConversationalRetrievalChain: Enables context-aware retrieval-augmented responses from the LLM."""

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
"""
GeneralAgentSARSA: Reinforcement learning-based agent that chooses actions 
and learns optimal behavior based on feedback using the SARSA algorithm.
"""


def get_response(user_input):
    """
        Generates a response to the given user input using the SARSA-powered agent.

        Args:
            user_input (str): The user's message or query.

        Returns:
            str: The assistant's response generated based on retrieval and LLM reasoning.
        """
    response = agent.process_input(user_input)
    return response


def update_feedback(feedback_type):
    """
        Updates the agent's SARSA Q-table based on user feedback.

        Args:
            feedback_type (str): Type of feedback, expected values: "positive" or "negative".

        Returns:
            None
        """
    reward = agent.handle_feedback(feedback_type.lower())
