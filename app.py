from flask import Flask, request, jsonify
import os
import csv
import requests
from io import BytesIO
import tempfile
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document

app = Flask(__name__)

# Initialize your OpenAI API key
API_KEY = 'sk-zeOdUFHCe5VinrJLmyGXcEbWy5RZo4mPFIH2KyAd2iT3BlbkFJhCjoGke12V9lWeK9RzzbToNKCgQTIUDavxE39ZJ94A'

# Initialize global variables
chroma_client = None
qa_chain = None

def load_documents_from_url(file_url):
    """Load documents from a deployed URL."""
    try:
        response = requests.get(file_url)
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.headers.get('Content-Type')}")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to download file: {response.status_code}")
        
        file_extension = os.path.splitext(file_url)[1].lower()

        if file_extension == ".txt":
            print("TXT File Content:", response.text[:500])  # Print the first 500 chars of the file
            return [Document(page_content=response.text)]
        elif file_extension == ".pdf":
            print("PDF File Detected")
            # Use a temporary file to save the PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(response.content)
                temp_pdf_path = temp_pdf.name
            # Load the PDF using PyPDFLoader with the file path
            loader = PyPDFLoader(temp_pdf_path)
            return loader.load()
        elif file_extension == ".csv":
            print("CSV File Detected")
            return load_csv_from_content(response.content)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    except Exception as e:
        print(f"Error loading document: {e}")
        raise

def load_csv_from_content(file_content):
    """Load and parse a CSV file from downloaded content."""
    documents = []
    content_str = file_content.decode('utf-8')
    reader = csv.reader(content_str.splitlines())
    for row in reader:
        # Create Document objects for each row
        documents.append(Document(page_content=" ".join(row)))
    return documents

# Initialize the OpenAI embedding model with the API key
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# Initialize Chroma vector store
chroma_client = Chroma(embedding_function=embeddings, persist_directory="./chroma_store")

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Initialize OpenAI LLM with the API key for chat models using ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=API_KEY)

@app.route('/post', methods=['POST'])
def ask_question():
    data = request.get_json()
    file_url = data.get('file_url', '')
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Load documents if file_url is provided
    documents = []
    if file_url:
        documents = load_documents_from_url(file_url)
    
    if documents:
        # Split documents into smaller chunks
        docs = text_splitter.split_documents(documents)
        # Add documents to Chroma vector store
        global chroma_client
        chroma_client.add_documents(docs)

    # Set up the Retrieval QA chain with the updated documents
    global qa_chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=chroma_client.as_retriever(),
        return_source_documents=True
    )

    response = qa_chain({"query": query})
    return jsonify({"result": response["result"]})

if __name__ == '__main__':
    app.run(debug=True)
