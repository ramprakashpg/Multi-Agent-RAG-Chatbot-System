import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000/process/"

def get_backend_response(prompt: str, mode: str):
    parameters = {"prompt": prompt, "parameter": mode}
    try:
        api_response = requests.post(BACKEND_URL, params=parameters)
        api_response.raise_for_status()

        data = api_response.json()
        return data.get("response", "Error: No 'response' key found in backend data.")

    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to the backend at {BACKEND_URL}. Is it running?"
    except requests.exceptions.RequestException as e:
        return f"Error: Backend request failed: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Streamlit UI
st.set_page_config(page_title="Multi-Chatbot", layout="wide")
st.title("🤖 Multi-Agent Chatbot")

st.sidebar.title("Select Chat")

def chat_selection_card(title, key):
    with st.sidebar.container():
        if st.button(title, key=f"chat_button_{key}"):
            st.session_state.chat_mode = title

chat_selection_card("General Chat", "general")
chat_selection_card("AI Specialist Chat", "ai")
chat_selection_card("Concordia Chat", "concordia")

# Initialize default chat mode
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "General Chat"

# Initialize chat history for each mode if not present in session state
if "general_chat_history" not in st.session_state:
    st.session_state.general_chat_history = [{"role": "assistant", "content": "Hi, I'm the general assistant. How can I help?"}]
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = [{"role": "assistant", "content": "Greetings! I'm the AI specialist. What's on your mind?"}]
if "concordia_chat_history" not in st.session_state:
    st.session_state.concordia_chat_history = [{"role": "assistant", "content": "Hello! I'm the Concordia expert. Ask me anything."}]

# Determine the current chat history based on the selected mode
if st.session_state.chat_mode == "General Chat":
    current_chat_history = st.session_state.general_chat_history
    agent_mode = "general"
elif st.session_state.chat_mode == "AI Specialist Chat":
    current_chat_history = st.session_state.ai_chat_history
    agent_mode = "ai"
elif st.session_state.chat_mode == "Concordia Chat":
    current_chat_history = st.session_state.concordia_chat_history
    agent_mode = "concordia"

# Display the current chat history
for message in current_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask the '{st.session_state.chat_mode.replace(' Chat', '')}' agent..."):
    current_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = get_backend_response(prompt, agent_mode)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response_text.split():
            full_response+= chunk + " "
            time.sleep(0.08)
            message_placeholder.markdown(full_response+ "▌")
        message_placeholder.markdown(response_text)

    current_chat_history.append({"role": "assistant", "content": response_text})

    # Update the session state with the new chat history
    if st.session_state.chat_mode == "General Chat":
        st.session_state.general_chat_history = current_chat_history
    elif st.session_state.chat_mode == "AI Specialist Chat":
        st.session_state.ai_chat_history = current_chat_history
    elif st.session_state.chat_mode == "Concordia Chat":
        st.session_state.concordia_chat_history = current_chat_history