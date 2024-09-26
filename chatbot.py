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

# Initialize Hugging Face model and token
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the LLM
llm = HuggingFaceHub(
    repo_id=hf_model,
    huggingfacehub_api_token=huggingface_token,
    model_kwargs={"temperature": 0.7}
)

# Load the tourism data from JSON file
json_file_path = "ALL_countries_document .json"  # Ensure this path is correct
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
embeddings_folder = os.path.join("embeddings_cache")  # Use relative path
os.makedirs(embeddings_folder, exist_ok=True)  # Create folder if it doesn't exist
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Create and save the vector store
vector_db_path = os.path.join("faiss_index")  # Use relative path
os.makedirs(vector_db_path, exist_ok=True)  # Create folder if it doesn't exist
vector_db = FAISS.from_documents(docs1, embeddings)
vector_db.save_local(vector_db_path)
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

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
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
        result = chain({"question": query})
        return result["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"


