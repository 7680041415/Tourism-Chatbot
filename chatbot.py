import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
import json
import os
from config import HUGGINGFACEHUB_API_TOKEN

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Load the LLM
from langchain_huggingface import HuggingFaceEndpoint
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# Update the JSON file path to your tourism data
json_file_path = "C:/Users/abhin/Desktop/Bootcamp/Final Project/ALL_countries_document .json"
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

# Set up embeddings and vector store
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "C:/Users/abhin/Desktop/Bootcamp/Final Project/embeddings_cache"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Create and save the vector store
vector_db_path = "C:/Users/abhin/Desktop/Bootcamp/Final Project/faiss_index2"
vector_db = FAISS.from_documents(docs1, embeddings)
vector_db.save_local(vector_db_path)
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
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
