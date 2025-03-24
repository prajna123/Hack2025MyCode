import streamlit as st
from custom_agent_IT3 import CustomAgentWrapperIT  # Import the wrapper
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

def pingGoogle():
    print("Ping Google")
    st.session_state["user_query"] = "Ping Google"
    print("Suggestion clicked", st.session_state.get("user_query"))

def runNetstat():
    print("Ping Google")
    st.session_state["user_query"] = "Run Netstat"
    print("Suggestion clicked", st.session_state.get("user_query"))

def checkIpconfig():
    print("Check ipconfig")
    st.session_state["user_query"] = "Check ipconfig"
    print("Suggestion clicked", st.session_state.get("user_query"))

def checkSplunk():
    st.session_state["user_query"] = "CheckSplunkDashboard"
    print("Suggestion clicked", st.session_state.get("user_query"))
    #print("Suggestion clicked for splunk_input_data", st.session_state.get("splunk_input_data"))
    



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

if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

if "splunk_input_data" not in st.session_state:
    st.session_state["splunk_input_data"] = {}
# User input
user_query = st.chat_input("Ask me anything...") or st.session_state.get("user_query")

if user_query:
    st.chat_message("user").write(user_query)
    print("User Query", user_query)

    # ✅ Call agent with vectorDB search
    if user_query in ["Open Google"] :
        print("Opening suggestions", user_query)
        print(f"Opening {user_query}")
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(OpenWebsite.open_website())
        st.chat_message("assistant").write(response)
        st.button('Check Google Services', on_click=checkGoogle)
        st.button('Ping Google Server', on_click=pingGoogle)
        st.button('Run netstat', on_click=runNetstat)
        st.button('Check ipconfig', on_click=checkIpconfig)
    elif "ping" in user_query.lower() or user_query in ["Ping Google"]:
        print("Ping suggestions came inside method", user_query)
        response = OpenWebsite.check_network_connectivity()
        st.chat_message("assistant").write(response)
        st.button('Check Google Services', on_click=checkGoogle)
        st.button('Ping Google Server', on_click=pingGoogle)
        st.button('Run netstat', on_click=runNetstat)
        st.button('Check ipconfig', on_click=checkIpconfig)
    elif "netstat" in user_query.lower() or user_query in ["Run Netstat"]:
        print("Ping suggestions came inside method", user_query)
        response = OpenWebsite.check_netstat()
        st.chat_message("assistant").write(response)
        st.button('Check Google Services', on_click=checkGoogle)
        st.button('Ping Google Server', on_click=pingGoogle)
        st.button('Run netstat', on_click=runNetstat)
        st.button('Check ipconfig', on_click=checkIpconfig)
    elif "ipconfig" in user_query.lower() or user_query in ["Check ipconfig"]:
        print("Ping suggestions came inside method", user_query)
        response = OpenWebsite.check_ipconfig()
        st.chat_message("assistant").write(response)
        st.button('Check Google Services', on_click=checkGoogle)
        st.button('Ping Google Server', on_click=pingGoogle)
        st.button('Run netstat', on_click=runNetstat)
        st.button('Check ipconfig', on_click=checkIpconfig)
    elif any(word in user_query.lower() for word in ["network", "internet"]):
        response = st.session_state.agent.query(user_query)
        st.chat_message("assistant").write(response)
        # ✅ Display Quick-Reply Buttons for Further Help
        st.write("Need more help? Try these:")
        st.button('Check Google Services', on_click=checkGoogle)
        st.button('Ping Google Server', on_click=pingGoogle)
        st.button('Run netstat', on_click=runNetstat)
        st.button('Check ipconfig', on_click=checkIpconfig)
    elif user_query == "CheckSplunkDashboard":
        print("Opening splunk suggestions", user_query)
        print(f"Fetching splunk input data {st.session_state.get("splunk_input_data")}")
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(OpenWebsite.search_splunk())
        st.chat_message("assistant").write(response)
    # elif any(word in user_query.lower() for word in ["splunk", "log", "event"]):
    #     print("Initial Splunk", user_query)
    #     response = st.session_state.agent.query(user_query)
    #     st.write("Try execute query in splunk dashboard:")
    #     fixed_keys = ["index", "host", "source", "sourcetype"]
    #     with st.form("user_input_form"):
    #         input_data = {}
    #         for key in fixed_keys:
    #             input_data[key] = st.text_input(f"{key}:", key=f"input_{key}")

    #         submitted = st.form_submit_button("Submit")
            
    #     if submitted:
    #         st.session_state["form_submitted"] = True
    #         st.session_state["splunk_input_data"] = input_data  # Store data
    #         st.experimental_rerun()
    elif any(word in user_query.lower() for word in ["splunk", "log", "event"]):
        print("Initial Splunk", user_query)
        response = st.session_state.agent.query(user_query)
        st.chat_message("assistant").write(response)
        st.write("Try execute query in splunk dashboard:")
        st.button('Check Splunk Services', on_click=checkSplunk)
        


    else:
        print("Generic Response")
        response = st.session_state.agent.query(user_query)
        st.chat_message("assistant").write(response)
    #print("User response", response)

    #st.chat_message("assistant").write(response)

    # Save chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

if st.session_state["form_submitted"]:
    st.write("✅ Form Submitted Successfully!")
    st.write(st.session_state["splunk_input_data"])  # Debugging
    print("🚀 Form Submitted:", st.session_state["splunk_input_data"])
    st.session_state["form_submitted"] = False