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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate


class CustomAgentWrapperIT:
    """Wrapper for managing a custom LLM-powered agent with VectorDB"""

    def __init__(self, api_key):
        """Initialize the custom agent with Network, splunk, microsoft documentation"""
        print("started**********")
        self.api_key = api_key
        print("api key", self.api_key)
        #self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=self.api_key)

        # ✅ Step 1: Load All Documentation
        loader1 = WebBaseLoader("https://support.microsoft.com/en-us")
        loader2 = WebBaseLoader("https://learn.microsoft.com/en-us/windows/release-health/status-windows-11-21h2")
        loader3 = WebBaseLoader("https://docs.citrix.com/")

        # Load documents from all loaders
        docs1 = loader1.load()
        docs2 = loader2.load()
        docs3 = loader3.load()

        text_loader1 = TextLoader("TroubleshootingDocs/Network_Troubleshooting_Guideline.txt")  # Use .doc for older formats
        #word_loader2 = UnstructuredFileLoader("TroubleshootingDocs/Splunk_Troubleshooting_Guideline.docx")
        text_docs1 = text_loader1.load()
        #word_docs2 = word_loader2.load()

        # Merge all documents into a single list
        self.docs = text_docs1

        documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(self.docs)

        # ✅ Step 3: Create VectorDB with FAISS
        vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())

        # ✅ Step 4: Define Retriever
        self.retriever = vectordb.as_retriever()
        retriever_tool=create_retriever_tool(self.retriever,"ips-faq-search","Search any information about Integrated Platform Support ")
        self.tools=[retriever_tool]

        # 🔹 Step 5: Define Prompt for Extracting Structured Steps
        self.prompt_template = PromptTemplate(
            template="""
                You are an AI assistant that extracts structured steps from a troubleshooting guide.
                Given the following document, return only a structured list of troubleshooting steps:

                {context}

                Format the response as:
                [
                    "Step 1",
                    "Step 2",
                    "Step 3",
                    ...
                ]
                """,
            input_variables=["context"],
        )

        # ✅ Step 5: Create Retrieval QA Chain
        # self.qa_chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=self.retriever
        # )

        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key="your_openai_api_key")
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=False)

        # ✅ Step 6: Add Conversation Memory (optional)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        #prompt=hub.pull("hwchase17/openai-functions-agent")

        self.agent = create_openai_functions_agent(
            ChatPromptTemplate.from_template(self.prompt_template),
            llm=self.llm,            # OpenAI language model
            tools=self.tools,        # VectorDB retriever tool
            agent="openai-tools",    # The correct agent type
            verbose=True,             # Enable logging for debugging
        )

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

    def classify_query(self, user_input):
        #llm = ChatOpenAI(model_name="gpt-4")  # Use your preferred model
        prompt = f"Classify the user query: '{user_input}'. Is it about extracting troubleshooting steps from a document? Answer with 'Yes' or 'No'."

        response = self.llm.invoke(prompt).strip().lower()
        return response == "yes"

    def query(self, user_input):
        """Executes query with both the agent and the vectorDB"""
        try:
            if self.classify_query(user_input):
                print("✅ This is a troubleshooting extraction query!")
                #qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), retriever=self.retriever)
                response = self.qa_chain.run(user_input)
                return response
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