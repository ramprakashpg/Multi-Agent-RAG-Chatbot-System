import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000/process/"
FEEDBACK_URL = "http://127.0.0.1:8000/feedback/"  # Replace with your feedback API URL

def get_backend_response(prompt: str, mode: str):
    payload = {"user_prompt": prompt, "agent": mode}
    try:
        api_response = requests.post(BACKEND_URL, json=payload)
        api_response.raise_for_status()
        data = api_response.json()
        return data.get("response", "Error: No 'response' key found in backend data.")
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to the backend at {BACKEND_URL}. Is it running?"
    except requests.exceptions.RequestException as e:
        return f"Error: Backend request failed: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def send_feedback(chat_mode: str, feedback_type: str):
    payload = {"agent": chat_mode, "feedback": feedback_type}
    try:
        feedback_response = requests.post(FEEDBACK_URL, json=payload)
        feedback_response.raise_for_status()
        st.success(f"Feedback sent: {feedback_type} in {chat_mode}")
    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to feedback API at {FEEDBACK_URL}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending feedback: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while sending feedback: {e}")

# Streamlit UI
st.set_page_config(page_title="Multi-Chatbot", layout="wide")

st.markdown("""
    <style>

    /* Style each radio label */
    div[role="radiogroup"] > label {
        border-radius: 8px;
        padding: 0.8em 1.2em;
        margin-bottom: 0.8em;
        cursor: pointer;
        border-left: 4px solid transparent;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        width: 100%;      /* Make all buttons take full width */
        box-sizing: border-box;  /* Include padding in width calculation */
    }

    /* Hover effect */
    div[role="radiogroup"] > label:hover {
        border: 2px solid #ffffff;
    }

    /* Selected item styling */
    div[role="radiogroup"] > label[data-selected="true"] {
        background-color: #d0e4ff !important;
        font-weight: bold;
        border-left: 4px solid #4285f4;
        box-shadow: 0 4px 8px rgba(66,133,244,0.2);
    }

    /* Icon styling */
    div[role="radiogroup"] > label::before {
        margin-right: 10px;
        font-size: 1.2em;
        flex-shrink: 0;  /* Prevent icon from shrinking */
    }
    
    /* Optional: If you want to set a specific fixed width instead of full width */
    /*
    div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    div[role="radiogroup"] > label {
        width: 300px;  /* Set your desired fixed width */
    }
    */
    </style>
""", unsafe_allow_html=True)

st.title("🤖 💬 Multi-Agent Virtual Assistant")

st.sidebar.title("Agent Center")

chat_mode = st.sidebar.radio(
    "",
    ("General Assistant", "AI Specialist", "Concordia HelpDesk"),
    index=0,
    key="chat_mode_selector"
)

# Initialize chat history for each mode if not present in session state
if "general_chat_history" not in st.session_state:
    st.session_state.general_chat_history = [{"role": "assistant", "content": "Hi, I'm the general assistant. How can I help?"}]
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = [{"role": "assistant", "content": "Greetings! I'm the AI specialist. What's on your mind?"}]
if "concordia_chat_history" not in st.session_state:
    st.session_state.concordia_chat_history = [{"role": "assistant", "content": "Hello! I'm the Concordia HelpDesk. Ask me anything."}]

# Determine the current chat history and agent mode
if chat_mode == "General Assistant":
    current_chat_history = st.session_state.general_chat_history
    agent_mode = "general"
elif chat_mode == "AI Specialist":
    current_chat_history = st.session_state.ai_chat_history
    agent_mode = "ai"
elif chat_mode == "Concordia HelpDesk":
    current_chat_history = st.session_state.concordia_chat_history
    agent_mode = "concordia"

# Display the current chat history with feedback buttons (conditional)
for i, message in enumerate(current_chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show feedback buttons only after the first user message
        # AND exclude the initial assistant message
        if (
            len(current_chat_history) > 1
            and message["role"] == "assistant"
            and i != 0  # Check if it's NOT the first message
        ):
            col1, col2, col3= st.columns([0.5, 0.5, 10])
            if col1.button("👍", key=f"thumbs_up_{chat_mode}_{i}"):
                send_feedback(chat_mode, "positive")
            if col2.button("👎", key=f"thumbs_down_{chat_mode}_{i}"):
                send_feedback(chat_mode, "negative")

if prompt := st.chat_input(f"Ask the '{chat_mode.replace(' Chat', '')}' agent..."):
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

        col1, col2, col3= st.columns([0.5, 0.5, 10])
        if col1.button("👍", key=f"thumbs_up_{chat_mode}_{len(current_chat_history) - 1}"):
            send_feedback(chat_mode, "positive")
        if col2.button("👎", key=f"thumbs_down_{chat_mode}_{len(current_chat_history) - 1}"):
            send_feedback(chat_mode, "negative")

    # Update the session state with the new chat history
    if chat_mode == "General Assistant":
        st.session_state.general_chat_history = current_chat_history
    elif chat_mode == "AI Specialist":
        st.session_state.ai_chat_history = current_chat_history
    elif chat_mode == "Concordia HelpDesk":
        st.session_state.concordia_chat_history = current_chat_history