import os
# from dotenv import load_dotenv # Uncomment if you need to load .env variables here

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_data():
    print('Starting document ingestion...')

    # Construct absolute path to the data directory
    # Assuming this script is run from the project root or smart_farming_recommender directory
    # For now, hardcode relative to the script's location
    app_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(app_dir, 'data')
    documents = []

    # Load documents from the data directory
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if filename.endswith('.pdf'):
            print(f'Loading PDF: {filename}...')
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith('.txt'):
            print(f'Loading Text: {filename}...')
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    
    if not documents:
        print('No PDF documents found in the data directory. Please add some PDFs to rag_core/data/')
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')

    # Create embeddings and store in Chroma
    # Using the specified multi-modal embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Persist the Chroma vector store
    persist_directory = os.path.join(app_dir, 'chroma_db')
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()

    print(f'Successfully ingested documents into Chroma DB at {persist_directory}')

if __name__ == '__main__':
    # load_dotenv() # Uncomment if you need to load .env variables here
    ingest_data()
