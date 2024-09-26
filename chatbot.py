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
from config import HUGGINGFACEHUB_API_TOKEN
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms import HuggingFaceHub




# Define the model
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Specify parameters directly
llm = HuggingFaceHub(
    repo_id=hf_model,
    huggingfacehub_api_token=huggingface_token,
     # Specify temperature directly
)


# Now llm can be used as the language model in Langchain


# Update the JSON file path to your tourism data
json_file_path = "ALL_countries_document .json"
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
docs_split = text_splitter.split_documents(documents)

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Set up embeddings and vector store
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = os.path.join("embeddings_cache")  # Use relative path
os.makedirs(embeddings_folder, exist_ok=True)  # Create folder if it doesn't exist

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Create and save the vector store
vector_db_path = os.path.join("faiss_index2")  # Use relative path
os.makedirs(vector_db_path, exist_ok=True)  # Create folder if it doesn't exist

# Create the vector store
vector_db = FAISS.from_documents(docs_split, embeddings)
vector_db.save_local(vector_db_path)

# Load the vector store
vector_db = FAISS.load_local(vector_db_path, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# Initialize memory for the chat
def init_memory():
    return ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

memory = init_memory()

# Update the prompt template to fit tourism context
template = """
You are a knowledgeable assistant with information about various countries and their tourism.
Answer the question based on the provided context below.

Previous conversation:
{chat_history}

Tourism context:
{context}

Question: {question}
Response:"""

from langchain.prompts.prompt import PromptTemplate

prompt = PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])

# Create the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# Function to ask questions to the chatbot
def ask_question(query):
    try:
        result = qa_chain({"question": query})
        return result["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"


