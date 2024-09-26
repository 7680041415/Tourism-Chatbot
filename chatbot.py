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
json_file_path = "ALL_countries_document.json"  # Ensure this path is correct
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

# Set up embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder="embeddings_cache")

# Create the FAISS vector store directly from documents
vector_store = FAISS.from_documents(docs_split, embeddings)

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

# Streamlit interface
st.title("Tourism Chatbot")
user_input = st.text_input("Ask a question about tourism:")
if user_input:
    response = ask_question(user_input)
    st.write(response)
