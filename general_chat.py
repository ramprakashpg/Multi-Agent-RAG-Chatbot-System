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

        # Load or initialize Q-table
        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path)
        else:
            self.q_table = np.zeros((100, len(actions)))  # Simplified state space

        # Store SARSA components
        self.current_state = None
        self.current_action = None
        self.next_state = None
        self.next_action = None
        self.awaiting_feedback = False

        # Track interaction history for debugging
        self.history = []

    def get_state(self, user_input):
        # Improved state representation - combines input length with content features
        # This is still simplified but better than just length
        state_features = [
            len(user_input) % 50,  # Length feature (0-49)
            sum(c.isalpha() for c in user_input) % 10,  # Character type feature
            1 if '?' in user_input else 0  # Question feature
        ]

        # Combine features into a single state index (still simplified)
        state_index = (state_features[0] * 20 + state_features[1] * 2 + state_features[2]) % 100
        return state_index

    def choose_action(self, state):
        # Epsilon-greedy policy with decaying epsilon for better convergence
        if random.uniform(0, 1) < max(0.01, self.epsilon):  # Epsilon floor of 0.01
            return random.choice(range(len(self.actions)))  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation

    def update_q_table(self, state, action, reward, next_state, next_action):
        # Standard SARSA update rule
        old_q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        self.q_table[state, action] = old_q_value + self.alpha * (reward + self.gamma * next_q_value - old_q_value)

        # Save Q-table periodically to persist learning
        if random.random() < 0.1:  # Save 10% of the time to avoid constant disk writes
            np.save(self.q_table_path, self.q_table)

        # Log update for debugging
        print(f"Updated Q({state},{action}) = {self.q_table[state, action]:.4f} (reward: {reward})")

    def process_input(self, user_input):
        # Get the next state based on the user input
        next_state = self.get_state(user_input)
        print(f"State: {next_state}")

        # Choose next action using epsilon-greedy policy
        next_action_idx = self.choose_action(next_state)
        next_action_name = self.actions[next_action_idx]
        print(f"Selected action: {next_action_name}")

        # If we have a previous state-action pair, update the Q-table with an immediate reward
        if self.awaiting_feedback and self.current_state is not None and self.current_action is not None:
            # Simple immediate reward: +0.1 for continuing the conversation
            immediate_reward = 0.1
            self.update_q_table(
                self.current_state,
                self.current_action,
                immediate_reward,
                next_state,
                next_action_idx
            )
            self.awaiting_feedback = False

        # Store current state-action for next update
        self.current_state = next_state
        self.current_action = next_action_idx
        self.awaiting_feedback = True

        # Add to history for tracking
        self.history.append((next_state, next_action_idx))

        # Use LangChain to generate response
        try:
            prompt_replaced = prompt_template.format(user_input=user_input, action=next_action_name)
            print(prompt_replaced)
            response = self.llm_chain.run(user_input)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "Sorry, I encountered an error while processing your request."

        return response

    def handle_feedback(self, feedback_type):
        # Process explicit feedback (thumbs up/down)
        if not self.awaiting_feedback or self.current_state is None or self.current_action is None:
            print("Warning: Cannot handle feedback without a preceding interaction.")
            return 0

        # Convert feedback to reward
        reward = 1.0 if feedback_type == "positive" else -1.0

        # We need a next state and action for SARSA
        # For explicit feedback, we'll use a special terminal-like state
        # This simplification works for feedback scenarios
        next_state = (self.current_state + 50) % 100  # Arbitrary next state
        next_action = np.argmax(self.q_table[next_state])  # Best action for that state

        # Update Q-table with the feedback
        self.update_q_table(
            self.current_state,
            self.current_action,
            reward,
            next_state,
            next_action
        )

        # Save Q-table after explicit feedback
        np.save(self.q_table_path, self.q_table)

        # Reset awaiting_feedback flag
        self.awaiting_feedback = False

        # Return the reward for external tracking
        return reward


# List of possible actions for the agent
actions = [
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
agent = GeneralAgentSARSA(actions, llm_chain)


def get_response(user_input):
    # Process the user input and get a response
    response = agent.process_input(user_input)
    return response

# For handling explicit feedback (thumbs-up and thumbs-down)
def update_feedback(feedback_type):
    reward = agent.handle_feedback(feedback_type.lower())