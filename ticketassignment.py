import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import re

load_dotenv() ## aloading all the environment variable

# open_api_key=os.getenv("OPENAI_API_KEY")
# client = openai.OpenAI(api_key=open_api_key)

GROQ_API_KEY = os.getenv("groq_api_key") 
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)


# Define agent mapping based on ticket categories
AGENT_MAPPING = {
    "Network Issue": "John Doe",
    "Splunk Issue": "Alice Smith",
    "Hardware Issue": "Bob Johnson",
    "other": "General Support"
}

# Function to classify ticket
# def classify_ticket(ticket_description):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an expert in IT ticket classification. Categorize the issue as 'network', 'splunk', 'hardware', or 'other'."},
#                 {"role": "user", "content": f"Ticket: {ticket_description}"}
#             ]
#         )
#         print("response is ", response)
#         category = response.choices[0].message.content.strip().lower()
#         print("choice is" , response.choices[0])
#         return category if category in AGENT_MAPPING else "other"
#     except Exception as e:
#         st.error(f"Error classifying ticket: {e}")
#         return "other"

def classify_ticket(ticket_description):
    prompt = f"""
    You are an AI-powered ticket classifier. 
    Analyze the following IT support ticket description and categorize it into one of the following categories:
    - Network Issue
    - Splunk Issue
    - Hardware Issue
    - other

    Ticket Description: {ticket_description}
    Category:
    """

    try:
        response = llm.invoke(prompt)
        raw_text = response.content.strip()
        print("raw_text is ", raw_text)

        # Extract category using regex (looks for the predefined categories)
        match = re.search(r"(Network Issue|Splunk Issue|Hardware Issue|other)", raw_text, re.IGNORECASE)
        category = match.group(1) if match else "Other"
        
        #category = response.content.strip()
        print("category is ", category)
        return category if category in AGENT_MAPPING else "Other"
    except Exception as e:
        return "Other"

# Function to process tickets
def process_tickets(df):
    df["category"] = df["ticket_description"].apply(classify_ticket)
    #df["assigned_agent"] = df["category"].apply(lambda x: AGENT_MAPPING.get(x, "General Support"))
    print("df[category] is",df["category"])
    df["assigned_agent"] = df["category"].map(AGENT_MAPPING) 
    print("df[assigned_agent] is",df["category"])
    return df

# Streamlit App UI
st.title("Automated Ticket Assignment System")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check if the required column exists
    if "ticket_description" not in df.columns:
        st.error("CSV must contain a 'ticket_description' column.")
    else:
        st.write("File uploaded successfully. Click 'Assign Tickets' to process.")

        # Button to process tickets
        if st.button("Assign Tickets"):
            with st.spinner("Processing tickets..."):
                df_processed = process_tickets(df)
                st.session_state["processed_tickets"] = df_processed
                st.success("Tickets assigned successfully!")

# Display results
if "processed_tickets" in st.session_state:
    st.write("### Assigned Tickets")
    st.dataframe(st.session_state["processed_tickets"])

    # Download option
    csv = st.session_state["processed_tickets"].to_csv(index=False)
    st.download_button("Download Assigned Tickets", data=csv, file_name="assigned_tickets.csv", mime="text/csv")