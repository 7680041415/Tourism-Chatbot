import os
import json
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hugging Face model and token (via Streamlit Cloud's Secrets Manager)
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
huggingface_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
if not huggingface_token:
    st.error("Hugging Face API token is missing.")
    st.stop()

# Initialize the LLM
llm = HuggingFaceHub(
    repo_id=hf_model,
    huggingfacehub_api_token=huggingface_token,
    model_kwargs={"temperature": 0.7}
)

# Load the tourism data from JSON file (Ensure the file exists in the GitHub repo)
json_file_path = os.path.join(os.path.dirname(__file__), "ALL_countries_document.json")  
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Convert the loaded JSON data into Document objects
documents = []
for item in data:  # Iterate over the list of dictionaries
    page_content = item["page_content"]  # Get the full page content
    url = item["metadata"]["url"]  # Get the URL
    document = Document(page_content=page_content, metadata={"url": url})  # Create a Document object
    documents.append(document)

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs1 = text_splitter.split_documents(documents)

# Set up embeddings and vector store
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
# No need for saving embedding cache in Streamlit Cloud, work in memory
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Initialize FAISS and use it in-memory (no need to save/load locally in Streamlit Cloud)
vector_db = FAISS.from_documents(docs1, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Initialize memory for the chat
def init_memory():
    return ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

# Check if memory exists in session state
if 'memory' not in st.session_state:
    st.session_state.memory = init_memory()

memory = st.session_state.memory

# Update the prompt template to fit tourism context
template = """
You are a knowledgeable assistant with information about various countries and their tourism.
Answer the question based on the provided context below.


Tourism context:
{context}

Question: {question}
Response:"""

prompt = PromptTemplate(template=template, input_variables=[ "context", "question"])

# Initialize the conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Function to ask questions to the chatbot
def ask_question(query):
    try:
        # Ensure chat history is retrieved properly
        result = chain({
            "question": query,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })
        logger.info(f"User question: {query}")
        logger.info(f"Response: {result['answer']}")
        return result["answer"]
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return "Sorry, I encountered an issue processing your request."

# Streamlit interface
st.title("Tourism Assistant")
query = st.text_input("Ask a question about tourism:")
if st.button("Submit"):
    if query:
        response = ask_question(query)
        st.write(response)
    else:
        st.warning("Please enter a question.")
