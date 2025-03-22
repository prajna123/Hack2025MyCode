import streamlit as st
from custom_agent_IT import CustomAgentWrapperIT  # Import the wrapper
from OpenWebsite import OpenWebsite
import os
from dotenv import load_dotenv
import asyncio

# Streamlit UI Setup
st.set_page_config(page_title="Integrated Platform Support AI Chatbot", layout="wide")

load_dotenv() ## aloading all the environment variable

open_api_key=os.getenv("OPENAI_API_KEY")


def checkGoogle():
    print("Checking Google")
    st.session_state["user_query"] = "Open Google"
    print("Suggestion clicked", st.session_state.get("user_query"))



# Sidebar: API Key Input
#openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

suggestions = ["Open Google", "Check Google Maps", "Read Google News", "ping Google"]
# ✅ Initialize agent (only once)
if "agent" not in st.session_state and open_api_key:
    st.session_state.agent = CustomAgentWrapperIT(api_key=open_api_key)

# Streamlit Chat UI
st.title("📄 Integrated Platform Support AI Chatbot")
st.write("Ask questions based on Platform Support tickets documentation.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

if "user_query" not in st.session_state:
    st.session_state["user_query"] = None
# User input
user_query = st.chat_input("Ask me anything...") or st.session_state.get("user_query")

if user_query:
    st.chat_message("user").write(user_query)
    print("User Query", user_query)

    # ✅ Call agent with vectorDB search
    if any(word in user_query.lower() for word in ["network", "internet"]):
        response = st.session_state.agent.query(user_query)

        # ✅ Display Quick-Reply Buttons for Further Help
        st.write("Need more help? Try these:")
        st.button('Check Google Services', on_click=checkGoogle)
    elif user_query in ["Open Google"] :
        print("Opening suggestions", user_query)
        print(f"Opening {user_query}")
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(OpenWebsite.open_website())
    
    elif "ping" in user_query.lower():
        response = OpenWebsite.check_network_connectivity()
    else:
        response = st.session_state.agent.query(user_query)
    print("User response", response)

    st.chat_message("assistant").write(response)

    # Save chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response})