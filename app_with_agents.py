# Import necessary libraries
import streamlit as st
from agents import Head_Agent  

# Set up the title of the application
st.title("Chatbot 🤖: Najib + Urvashi")

# Initialize API keys
open_ai_api_key = st.secrets["OPEN_AI_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = "miniproject-2"

# Initialize Head Agent
head_agent = Head_Agent(
    openai_key=open_ai_api_key,
    pinecone_key=pinecone_api_key,
    pinecone_index_name=index_name
)

# Initialize session state variables if not already set
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define a function to get the conversation history
def get_conversation() -> str:
    """
    Returns a formatted string of the conversation history.
    """
    conversation = ""
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        conversation += f"{role}: {message['content']}\n"
    return conversation

# Wait for user input
if user_input := st.chat_input("What would you like to chat about?"):
    # Append user input to messages
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response using Head Agent
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder to stream response

        # Use head_agent's main_loop to process the query
        ai_reply = head_agent.main_loop(user_input)

        # Display AI's response
        response_placeholder.markdown(ai_reply)

    # Append AI response to messages
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
