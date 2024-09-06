from flask import Flask, request, jsonify
import os
import csv
import requests
import tempfile
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

app = Flask(__name__)

# Initialize your OpenAI API key
API_KEY = ''  # Add your API key here

def load_documents_from_url(file_url):
    """Load documents from a URL (supporting txt, pdf, csv)."""
    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download file: {response.status_code}")
        
        file_extension = os.path.splitext(file_url)[1].lower()

        if file_extension == ".txt":
            # For .txt files, return the text content as Document objects
            return [Document(page_content=response.text)]
        elif file_extension == ".pdf":
            # For .pdf files, save the PDF to a temporary file and load it using PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(response.content)
                temp_pdf_path = temp_pdf.name
            loader = PyPDFLoader(temp_pdf_path)
            return loader.load()
        elif file_extension == ".csv":
            # For .csv files, parse the CSV and return the rows as Document objects
            return load_csv_from_content(response.content)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    except Exception as e:
        print(f"Error loading document: {e}")
        raise

def load_csv_from_content(file_content):
    """Load and parse CSV file content and convert it into Document objects."""
    documents = []
    content_str = file_content.decode('utf-8')
    reader = csv.reader(content_str.splitlines())
    for row in reader:
        # Convert each row into a space-separated string and wrap in a Document
        documents.append(Document(page_content=" ".join(row)))
    return documents

# Initialize the OpenAI embedding model with the API key
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# Initialize Chroma vector store
chroma_client = Chroma(embedding_function=embeddings, persist_directory="./chroma_store")

# Initialize OpenAI LLM with the API key for chat models using ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=API_KEY)

@app.route('/post', methods=['POST'])
def ask_question():
    data = request.get_json()
    file_url = data.get('file_url')
    query = data.get('query')

    if not file_url:
        return jsonify({'error': 'No file URL provided'}), 400
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Load and process documents from the file URL
        documents = load_documents_from_url(file_url)

        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Add documents to Chroma vector store
        chroma_client.add_documents(docs)

        # Set up a Retrieval QA chain with OpenAI and Chroma
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=chroma_client.as_retriever(),
            return_source_documents=True
        )

        # Get the answer to the query
        response = qa_chain({"query": query})

        # Serialize the result and the source documents
        result = {
            "result": response["result"],
            # "source_documents": [
            #     {
            #         "page_content": doc.page_content,  # Extract relevant content from Document object
            #     } for doc in response["source_documents"]
            # ]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
