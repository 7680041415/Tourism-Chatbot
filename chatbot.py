import os
import json
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from config import HUGGINGFACEHUB_API_TOKEN
from langchain import HuggingFaceHub

# Hugging Face Model and Token Setup
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the language model
llm = HuggingFaceHub(
    repo_id=hf_model,
    huggingfacehub_api_token=huggingface_token,
    model_kwargs={"temperature": 0.7}  # Optional: Set other parameters like temperature
)

# Load the tourism data from JSON file
json_file_path = "ALL_countries_document.json"  # Adjust if needed
with open(json_file_path, 'r') as file:
    data1 = json.load(file)

# Convert the loaded JSON data into Document objects
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
def init_memory():
    return ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

memory = init_memory()

# Update the prompt template to fit the tourism context
template = """
You are a knowledgeable assistant with information about various countries and their tourism.
Answer the question based on the provided context below.

Previous conversation:
{chat_history}

Tourism context:
{context}

Question: {question}
Response:
"""

# Ensure input variables match the prompt
prompt = PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])

# Create the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Function to ask questions to the chatbot
def ask_question(query):
    try:
        result = qa_chain({"question": query})
        return result["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

