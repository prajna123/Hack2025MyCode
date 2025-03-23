from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import logging
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub
import pandas as pd
import glob
from langchain.document_loaders import WebBaseLoader, UnstructuredFileLoader

class CustomAgentWrapperIT:
    """Wrapper for managing a custom LLM-powered agent with VectorDB"""

    def __init__(self, api_key):
        """Initialize the custom agent with Network, splunk, microsoft documentation"""
        print("started**********")
        self.api_key = api_key
        print("api key", self.api_key)
        self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=self.api_key)

        # ✅ Step 1: Load All Documentation
        loader1 = WebBaseLoader("https://support.microsoft.com/en-us")
        loader2 = WebBaseLoader("https://learn.microsoft.com/en-us/windows/release-health/status-windows-11-21h2")
        loader3 = WebBaseLoader("https://docs.citrix.com/")

        # Load documents from all loaders
        docs1 = loader1.load()
        docs2 = loader2.load()
        docs3 = loader3.load()

        word_loader1 = UnstructuredFileLoader("TroubleshootingDocs/Network_Troubleshooting_Guideline.docx")  # Use .doc for older formats
        word_loader2 = UnstructuredFileLoader("TroubleshootingDocs/Splunk_Troubleshooting_Guideline.docx")
        word_docs1 = word_loader1.load()
        word_docs2 = word_loader2.load()

        # Merge all documents into a single list
        docs = docs1 + docs2 + docs3 + word_docs1 + word_docs2

        documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

        # ✅ Step 3: Create VectorDB with FAISS
        vectordbmongo = FAISS.from_documents(documents, OpenAIEmbeddings())

        # ✅ Step 4: Define Retriever
        self.retriever = vectordbmongo.as_retriever()
        retriever_tool=create_retriever_tool(self.retriever,"mongo-faq-search","Search any information about Integrated Platform Support ")
        self.tools=[retriever_tool]

        # ✅ Step 5: Create Retrieval QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )

        # ✅ Step 6: Add Conversation Memory (optional)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        prompt=hub.pull("hwchase17/openai-functions-agent")

        self.agent=create_openai_tools_agent(self.llm,self.tools,prompt)

        # ✅ Step 7: Create Agent
        self.agent_executor=AgentExecutor(agent=self.agent,tools=self.tools,verbose=True)
        # self.agent_executor = initialize_agent(
        #     tools=[retriever_tool],  # Can add tools here
        #     llm=self.llm,
        #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     verbose=True,
        #     memory=self.memory,
        #     handle_parsing_errors=True,
        #     force_tool=True 
        # )

    def query(self, user_input):
        """Executes query with both the agent and the vectorDB"""
        try:
            # First, search from vectorDB
            rag_response = self.qa_chain.run(user_input)
            print("rag_response", rag_response) 

            # Then, use agent for reasoning (if needed)
            agent_response = self.agent_executor.invoke({"input": user_input})
            print("agent_response", agent_response)

          #  return f"🔍 **IT Docs:** {rag_response}\n🤖 **AI Reasoning:** {agent_response}"
            return f"🔍 **IT Docs:** {agent_response}"
        except Exception as e:
            logging.error(f"Agent error: {e}")
            return "Sorry, an error occurred while processing your request."

    def clear_memory(self):
        """Clears conversation memory"""
        self.memory.clear()