import os
import pytesseract
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_pdf_documents(directory):
    """Load and extract text from PDF files in the given directory."""
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file}...")
            file_path = os.path.join(directory, pdf_file)
            
            # Extract text from PDF
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Create a Document object
            doc = Document(
                page_content=text,
                metadata={"source": pdf_file}
            )
            documents.append(doc)
            print(f"Successfully processed {pdf_file}")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return documents

def main():
    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF files here.")
        return

    # Ensure the vector database directory exists
    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")
        print("Created 'vector_db_dir' directory for storing vectorized documents.")

    try:
        # Load the embedding model
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings()
        
        # Load and process PDF documents
        print("Loading and processing PDF documents...")
        documents = load_pdf_documents("data")
        
        if not documents:
            print("No documents were successfully processed. Please check your PDF files.")
            return
            
        print(f"Successfully loaded {len(documents)} documents")

        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(text_chunks)} chunks")

        # Create and persist the vector database
        print("Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )
        
        print("Successfully vectorized and stored documents in 'vector_db_dir'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure all required dependencies are installed:")
        print("pip install langchain langchain-community langchain-chroma langchain-huggingface PyPDF2 pytesseract")

if __name__ == "__main__":
    main()
