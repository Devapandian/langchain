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
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv('API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=API_KEY)

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
            print("TXT File Content:", response.text[:500])  
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

@app.route('/post', methods=['POST'])
def ask_question():
    data = request.get_json()
    file_url = data.get('file_url', '')
    query = data.get('query', '')
    username = data.get('username', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if not username:
        return jsonify({'error': 'No username provided'}), 400

    # Check if the query is a request for file summary
    file_summary_request = "summary of the file" if "summary" in query.lower() else ""

    # Construct the appropriate prompt based on the query or summary request
    wealth_manager_prompt = """
    You are a highly experienced wealth manager providing personalized financial advice.
    The user you're interacting with is {username}. They have provided a file and a question for you to assist with.
    
    If the user is requesting a summary of the file, provide a concise summary. 
    If a query is included, respond to the query with relevant financial advice based on the user's financial situation, goals, and wealth management interests.

    Summary request: {file_summary_request}
    Question: {query}

    Output Format:
    Provide the output as a list of JSON objects with these fields: "summary", "question", "answer", "username".
    REMEMBER: If there is no relevant information within the context, just say "I don’t have enough information, I'm sorry". Don't try to make up an answer. 
    This is not a suggestion. This is a rule.
    Don't answer questions about general culture or knowledge outside wealth management or the context provided by the user.
    """
    
    if file_summary_request:
        prompt = wealth_manager_prompt.format(
            username=username,
            file_summary_request=file_summary_request,
            query=""
        )
    else:
        prompt = wealth_manager_prompt.format(
            username=username,
            file_summary_request="",
            query=query
        )

    # Define the Chroma vector store directory for the user
    user_persist_directory = f"./chroma_store/{username}"
    os.makedirs(user_persist_directory, exist_ok=True)

    # Initialize Chroma vector store for the user
    chroma_client = Chroma(embedding_function=embeddings, persist_directory=user_persist_directory)

    documents = []
    if file_url:
        documents = load_documents_from_url(file_url)
    
    if documents:
        docs = text_splitter.split_documents(documents)
        chroma_client.add_documents(docs)

    # Set up the Retrieval QA chain with the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=chroma_client.as_retriever(),
        return_source_documents=True
    )

    # Use the custom wealth management prompt
    response = qa_chain({"query": prompt})

    result = response["result"]

    # Format response based on whether it's a summary or a query answer
    if file_summary_request:
        return jsonify({
            "summary": result,
            "question": query,
            "username": username
        })
    else:
        return jsonify({
            "question": query,
            "answer": result,
            "username": username
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
