# Import necessary libraries
import streamlit as st
from openai import OpenAI  # Ensure you install OpenAI: pip install openai

# Set up the title of the application
st.title("Chatbot 🤖: Najib + Urvashi")

api_key = st.secrets["OPEN_AI_KEY"]
client = OpenAI(api_key=api_key)


# Define a function to get the conversation history (Useful for Part-3)
def get_conversation() -> str:
    """
    Returns a formatted string of the conversation history.
    """
    conversation = ""
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        conversation += f"{role}: {message['content']}\n"
    return conversation

# Initialize session state variables if not already set
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if user_input := st.chat_input("What would you like to chat about?"):
    # Append user input to messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder to stream response
        response = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=st.session_state.messages
        )
        
        # Extract AI's reply
        ai_reply = response.choices[0].message.content
        response_placeholder.markdown(ai_reply)  # Display AI's response

    # Append AI response to messages
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})