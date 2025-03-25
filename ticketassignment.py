import openai
import pandas as pd
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from tqdm import tqdm
import streamlit as st
from dotenv import load_dotenv
import re

# 🔹 Set OpenAI API Key
load_dotenv() ## aloading all the environment variable
open_api_key=os.getenv("OPENAI_API_KEY")

# 🔹 Streamlit file uploader
st.title("Upload Ticket Data CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

AGENT_MAPPING = {
    ("Network Issue", "Critical", "Open", "Escalated"): "John Doe",
    ("Network Issue", "Critical", "Open", "Escalated"): "John Force",
    ("Network Issue", "High", "In Progress", "Not Escalated"): "Tom Hans",
    ("Network Issue", "High", "Open", "Not Escalated"): "Tom Hans",
    ("Network Issue", "Medium", "Open", "Not Escalated"): "Terry Frank",
    ("Network Issue", "Medium", "In Progress", "Not Escalated"): "Terry Frank",
    ("Network Issue", "Low", "Open", "Not Escalated"): "Scott Johnson",
    
    ("Splunk Issue", "Critical", "Open", "Escalated"): "Alice Smith",
    ("Splunk Issue", "High", "Open", "Not Escalated"): "Merry Jane",
    ("Splunk Issue", "Medium", "Open", "Not Escalated"): "Denise Joe",
    ("Splunk Issue", "Low", "Open", "Not Escalated"): "Rebecca Horis",

    ("Hardware Issue", "Critical", "Open", "Escalated"): "Bob Johnson",
    ("Hardware Issue", "High", "Open", "Not Escalated"): "Bob Johnson",
    ("Hardware Issue", "Medium", "Open", "Not Escalated"): "Dessei Fen",
    ("Hardware Issue", "Low", "Open", "Not Escalated"): "Robert Biss",
    
    ("Other", "High"): "Escalation Team",
    ("Other", "Medium"): "General Support",
    ("Other", "Low"): "Helpdesk Staff",
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ File Uploaded Successfully")

    # 🔹 Ensure required columns exist
    required_columns = ["ticket_description", "priority", "status", "customer_escalation"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 🔹 Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 🔹 Split tickets into chunks for vector storage
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = [Document(page_content=row["ticket_description"], metadata={"priority": row["priority"], "status": row["status"]}) for _, row in df.iterrows()]
    chunks = text_splitter.split_documents(documents)

    # 🔹 Create FAISS vector store
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    # 🔹 Define OpenAI LLM for classification
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # 🔹 Create RAG-based QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    def assign_agent(category, priority, status, customer_escalation):
        """Assign an agent based on category, priority, and status."""
        if status.lower() == "in progress":
            return "Already Assigned"
    
        print("category is", category)
        print("priority is", priority)
        print("status is", status)
        print("customer_escalation is", customer_escalation)
        return AGENT_MAPPING.get((category, priority, status, customer_escalation), "General Support")

    # 🔹 Ticket Classification Function
    def classify_ticket(ticket_description, priority, status, customer_escalation):
        try:
            query = f"""
            Given the following IT support ticket, classify it into one of these categories:
            - Network Issue
            - Hardware Issue
            - Software Issue
            - Other

            Consider the priority and status for better classification.

            **Ticket Details:**
            - Description: {ticket_description}
            - Priority: {priority}
            - Status: {status}
            - Customer Escalation: {customer_escalation}

            What category does this ticket fall into?
            """
            
            # 🔍 Retrieve relevant past tickets
            retrieved_tickets = retriever.get_relevant_documents(ticket_description)
            
            # 📌 Prepare context for OpenAI
            context = "\n".join([doc.page_content for doc in retrieved_tickets])
            # Construct final prompt properly
            final_prompt = f"""
            Past Tickets:
            {context}
            
            New Ticket:
            {query}
            """
            
            # 🤖 Get response from OpenAI
            response = llm.invoke(final_prompt)
            raw_text = response.content.strip()
            print("raw_text is ", raw_text)
            match = re.search(r"(Network Issue|Splunk Issue|Hardware Issue|other)", raw_text, re.IGNORECASE)
            category = match.group(1) if match else "Other"
        
            #category = response.content.strip()
            print("category is ", category)
            return category
        except Exception as e:
            print(f"Error classifying ticket: {e}")
            return "Other"

    # 🔹 Ensure DataFrame is not empty
    if df.empty:
        raise ValueError("DataFrame is empty! Check if CSV is loaded properly.")

    # 🔹 Add category column (if missing)
    if "category" not in df.columns:
        df["category"] = ""

    # 🔹 Apply ticket classification with progress tracking
    tqdm.pandas()
    st.write("File uploaded successfully. Click 'Assign Tickets' to process.")

    # Button to process tickets
    if st.button("Assign Tickets"):
        with st.spinner("Processing tickets..."):
            df["category"] = df.progress_apply(lambda x: classify_ticket(x["ticket_description"], x["priority"], x["status"], x["customer_escalation"]), axis=1)
            df["Assigned Agent"] = df.apply(lambda x: assign_agent(x["category"], x["priority"], x["status"], x["customer_escalation"]), axis=1)
            st.session_state["processed_tickets"] = df
            st.success("Tickets assigned successfully!")
    

    # 🔹 Save updated file
    if "processed_tickets" in st.session_state:
        st.write("### Assigned Tickets")
        st.dataframe(st.session_state["processed_tickets"])

        # Download option
        csv = st.session_state["processed_tickets"].to_csv(index=False)
        st.download_button("Download Assigned Tickets", data=csv, file_name="assigned_tickets.csv", mime="text/csv")
