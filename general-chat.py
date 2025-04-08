import numpy as np
import random
import os

from langchain.memory import ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

llm = OllamaLLM(model="llama3.2")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory="./chroma_general", embedding_function=embedding_model)
retriever = vector_store.as_retriever()

# LangChain prompt template
prompt_template = "User input: {user_input}\nAgent action: {action}\nrespond accordingly."

# Create LangChain's ConversationRetrievalChain
llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)


class GeneralAgentSARSA:
    def __init__(self, actions, llm_chain, alpha=0.1, gamma=0.9, epsilon=0.1, q_table_path="SARSA_q_table.npy"):
        self.actions = actions  # List of possible actions
        self.llm_chain = llm_chain  # LangChain LLM chain
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table_path = q_table_path

        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path)
        else:
            self.q_table = np.zeros((100, len(actions)))  # Simplified state space

        # Store the last state and action for feedback updates
        self.last_state = None
        self.last_action = None
        self.last_next_state = None
        self.last_next_action = None

    def get_state(self, user_input):
        # For simplicity, we represent state as the length of the user input.
        return len(user_input) % 100  # Simple state based on the input length modulo 100

    def choose_action(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.actions)))  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation

    def update_q_table(self, state, action, reward, next_state, next_action):
        # Apply SARSA update rule
        old_q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        self.q_table[state, action] = old_q_value + self.alpha * (reward + self.gamma * next_q_value - old_q_value)

    def process_input(self, user_input):
        # Get the current state
        state = self.get_state(user_input)

        # Choose an action using epsilon-greedy policy
        action_idx = self.choose_action(state)
        action_name = self.actions[action_idx]

        # Use LangChain's ConversationRetrievalChain
        try:
            prompt_replaced = prompt_template.format(user_input=user_input, action=action_name)
            print(prompt_replaced)
            response = self.llm_chain.run(prompt_replaced)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "Sorry, I encountered an error while processing your request."

        # For SARSA, we need the next state and action, but we don't have those yet
        # We'll store the current state and action for now
        self.last_state = state
        self.last_action = action_idx

        return response

    def handle_feedback(self, feedback):
        # We can't update the Q-table immediately since we don't know the next state/action yet
        # This will be called after we get the next user input and have chosen the next action
        if self.last_state is not None and self.last_action is not None and self.last_next_state is not None and self.last_next_action is not None:
            # Feedback can be thumbs-up (+1) or thumbs-down (-1)
            reward = 1 if feedback == "thumbs-up" else -1

            # Update the Q-table with the received feedback and chosen action
            self.update_q_table(self.last_state, self.last_action, reward, self.last_next_state, self.last_next_action)

            np.save(self.q_table_path, self.q_table)

            # Reset last_state and last_action after updating
            self.last_state = None
            self.last_action = None

            return reward
        else:
            print("Warning: Cannot handle feedback without previous state and action.")
            return 0

    def prepare_for_next_input(self, next_user_input):
        # This should be called after getting the next user input but before processing it
        if self.last_state is not None and self.last_action is not None:
            next_state = self.get_state(next_user_input)
            next_action_idx = self.choose_action(next_state)

            self.last_next_state = next_state
            self.last_next_action = next_action_idx


# List of possible actions for the agent
actions = [
    "Casual Conversation",
    "Information Retrieval",
    "Acknowledgment",
    "Small Talk",
    "Compliment/Encouragement"
]

# Create the agent with SARSA, LangChain, and ChromaDB
agent = GeneralAgentSARSA(actions, llm_chain)

def get_response(user_input):
    response = agent.process_input(user_input)
    agent.prepare_for_next_input(user_input)

    return response

# thumbs-up and thumbs-down
def update_feedback(feedback_type):
    reward = agent.handle_feedback(feedback_type.lower())
