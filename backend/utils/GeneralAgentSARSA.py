import os
import random

import numpy as np
from openpyxl.reader.excel import load_workbook
from openpyxl.workbook import Workbook

prompt_template = "User input: {user_input}\nAgent action: {action}\nrespond accordingly."


class GeneralAgentSARSA:
    def __init__(self, actions, llm_chain, alpha=0.1, gamma=0.9, epsilon=0.1, q_table_path="",
                 log_file=""):
        self.actions = actions  # List of possible actions
        self.llm_chain = llm_chain  # LangChain LLM chain
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table_path = q_table_path
        self.log_file = log_file

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

    def add_query_response(self, query='', response='', feedback=0.0):
        # Load the existing workbook
        wb = load_workbook(self.log_file)
        ws = wb.active

        # Append a new row with the query and response
        ws.append([query, response, feedback])  # Leave the 'Feedback' column empty initially
        wb.save(self.log_file)
        print(f"Added query and response: {query}, {response}")

    def add_feedback_to_last_row(self, feedback=0.0):
        # Load the existing workbook
        wb = load_workbook(self.log_file)
        print(self.log_file, self.q_table_path)
        ws = wb.active

        # Find the last row
        last_row = ws.max_row

        # Add feedback to the last row (Feedback is in column 3)
        ws.cell(row=last_row, column=3, value=feedback)
        print("Added feedback...")
        wb.save(self.log_file)
        print(f"Feedback added to the last row: {feedback}")

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
        # Save 10% of the time to avoid constant disk writes
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
            if not os.path.exists(self.log_file):
                # If not, create a new workbook and add headers for Query, Response, and Feedback
                wb = Workbook()
                ws = wb.active
                ws.append(['Query', 'Response', 'Feedback'])  # Add headers
                wb.save(self.log_file)
                print(f"Created new Excel file: {self.log_file}")
            else:
                print(f"Excel file '{self.log_file}' already exists.")
            self.add_query_response(query=user_input, response=response)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "Sorry, I encountered an error while processing your request. Please try again!!"

        return response

    def handle_feedback(self, feedback_type):
        # Process explicit feedback (thumbs up/down)
        if not self.awaiting_feedback or self.current_state is None or self.current_action is None:
            print(self.log_file, self.q_table_path)
            print("Warning: Cannot handle feedback without a preceding interaction.")
            return 0

        # Convert feedback to reward
        reward = 1.0 if feedback_type == "positive" else -1.0
        self.add_feedback_to_last_row(reward)

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
