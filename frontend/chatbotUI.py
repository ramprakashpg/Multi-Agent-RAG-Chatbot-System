import streamlit as st
import requests
BACKEND_URL = "http://127.0.0.1:8000/process/"

#Backend endpoint for processing the prompt
def get_backend_response(prompt: str, mode: str):
    paramters = {"prompt": prompt, "parameter": mode}
    try:
        api_response = requests.post(BACKEND_URL, params=paramters)
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
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("🤖 Adaptive Multi-Agent Chatbot")

# Default message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assisstant",
            "content": "Hi there! Which agent do you want?"
        }
    ]

st.sidebar.title("Configuration")

chatbot_mode = st.sidebar.selectbox(
    "Select Agent:",
    ("general", "ai","concordia"),
    index = 0
)

# Display the messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask the '{chatbot_mode}' agent..."):
    st.session_state.messages.append({"role": "user:", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking using '{chatbot_mode} mode..."):
            response_text = get_backend_response(prompt, chatbot_mode)
            st.markdown(response_text)

    
    st.session_state.messages.append({"role": "assistant", "content": response_text})


