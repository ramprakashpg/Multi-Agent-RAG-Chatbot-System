import streamlit as st

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("🤖 Adaptive Multi-Agent Chatbot")

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