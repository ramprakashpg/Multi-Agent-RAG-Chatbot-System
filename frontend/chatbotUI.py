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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask the '{chatbot_mode}' agent..."):
    st.session_state.messages.append({"role": "user:", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)