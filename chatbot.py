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

# Ensure Hugging Face Model and Token Setup
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the LLM model from Hugging Face
llm = HuggingFaceHub(
    repo_id=hf_model,
    huggingfacehub_api_token=huggingface_token,
    model_kwargs={"temperature": 0.7}
)

# Load the tourism data from the JSON file
json_file_path = "ALL_countries_document .json"  # Adjust the path if needed
with open(json_file_path, 'r') as file:
    data1 = json.load(file)

# Convert loaded JSON data into Document objects
documents = []
for item in data1:  # Iterate over the list of dictionaries
    url = item.get("URL", "")  # Get the URL
    content = item.get("Content", "")  # Get the content
    document_content = f"URL: {url}\nContent: {content}"  # Combine them as the document content
    metadata = {"url": url}  # Include the URL as metadata
    documents.append(Document(page_content=document_content, metadata=metadata))

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs1 = text_splitter.split_documents(documents)

# Set up embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "embeddings_cache"  # Local cache folder
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Create the FAISS vector store directly from documents
vector_store = FAISS.from_documents(docs1, embeddings)

# Initialize memory for the chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    verbose=True
)

# Function to ask questions to the chatbot
def ask_question(query):
    try:
        result = qa_chain({"question": query})  # Ensure that the input is in the expected format
        return result.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        return f"An error occurred: {str(e)}"
