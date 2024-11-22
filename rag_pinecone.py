import os
import logging
import zipfile
from typing import List, Tuple
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables and API key
GROQ_API_KEY = "API_KEY"
PINECONE_API_KEY = "API_KEY"
COHERE_API_KEY = "API_KEY"
LANGSMITH_API_KEY = "API_KEY"

def setup_environment():
    """Set up environment variables and initialize Pinecone client."""
    os.environ.update({
        'GROQ_API_KEY': GROQ_API_KEY,
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'COHERE_API_KEY': COHERE_API_KEY,
        'LANGSMITH_API_KEY': LANGSMITH_API_KEY,
        'LANGSMITH_TRACING': 'true'
    })
    return Pinecone(api_key=PINECONE_API_KEY)

def extract_zip(zip_file_path: str, extract_folder_path: str):
    """Extract ZIP file contents to specified folder."""
    try:
        os.makedirs(extract_folder_path, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder_path)
        logging.info(f"Files extracted to: {extract_folder_path}")
    except Exception as e:
        logging.error(f"Error extracting ZIP file: {e}")
        raise

def extract_tables_with_references_row_wise(pdf_path: str, page_num: int) -> List[dict]:
    """Extract tables with column and row references, chunking row-wise and including column names."""
    row_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            
            for idx, table in enumerate(tables):
                if table and len(table) > 0:
                    column_headers = table[0]
                    for row_num, row in enumerate(table[1:], start=1):
                        row_text = f"Row {row_num} | " + " | ".join(f"{col}: {cell}" for col, cell in zip(column_headers, row) if cell)
                        row_chunks.append({
                            "page_num": page_num + 1,
                            "table_idx": idx + 1,
                            "row_num": row_num,
                            "row_text": row_text
                        })
    except Exception as e:
        logging.error(f"Error extracting tables from page {page_num + 1} of {pdf_path}: {e}")
    return row_chunks

def extract_text_from_page_with_fallback(pdf_reader, page_num: int, pdf_path: str) -> str:
    """Extract text from a single page with OCR fallback and table extraction."""
    page = pdf_reader.pages[page_num]
    text = page.extract_text()

    if not text or len(text.strip()) < 20:
        logging.info(f"Using OCR for page {page_num+1} of {os.path.basename(pdf_path)}")
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        ocr_text = pytesseract.image_to_string(images[0])
        text = f"[OCR]\n{ocr_text}"

    table_chunks = extract_tables_with_references_row_wise(pdf_path, page_num)
    for chunk in table_chunks:
        text += f"\n--- Table {chunk['table_idx']} on Page {chunk['page_num']} ---\n"
        text += chunk['row_text'] + "\n"

    return text

def process_pdfs(folder_path: str) -> Tuple[List[str], List[str]]:
    """Process PDFs to extract text and maintain source filenames."""
    texts, sources = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    full_text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page_text = extract_text_from_page_with_fallback(pdf_reader, page_num, pdf_path)
                        full_text += f"\nPage {page_num + 1}:\n{page_text}"
                    
                    if full_text.strip():
                        texts.append(full_text)
                        sources.append(filename)
                        logging.info(f"Successfully processed: {filename}")
                    else:
                        logging.warning(f"Skipping empty file: {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
    return texts, sources

def split_texts_with_sources(texts: List[str], sources: List[str]) -> List[Document]:
    """Split texts into chunks while maintaining source information."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    
    for text, source in zip(texts, sources):
        chunks = text_splitter.split_text(text)
        documents.extend([
            Document(page_content=chunk, metadata={"source": source})
            for chunk in chunks
        ])
    
    logging.info(f"Created {len(documents)} chunks from {len(texts)} documents")
    return documents

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    """Create Pinecone index if it doesn't exist."""
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logging.info(f"Created new index: {index_name}")
        else:
            logging.info(f"Using existing index: {index_name}")
    except Exception as e:
        logging.error(f"Error creating Pinecone index: {e}")
        raise

def create_vectorstore(documents: List[Document], index_name: str) -> PineconeVectorStore:
    """Create vector store from documents."""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vectorstore = PineconeVectorStore.from_documents(
            documents,
            embedding_model,
            index_name=index_name
        )
        logging.info(f"Vector store created with {len(documents)} documents")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise

def main():
    """Main function to orchestrate the RAG setup process."""
    try:
        # Initialize
        zip_file_path = 'path/to/documents'
        extract_folder_path = 'path/to/documents'
        index_name = "legal-rag"
        
        # Setup environment and Pinecone client
        pc = setup_environment()
        
        # Extract ZIP file
        #extract_zip(zip_file_path, extract_folder_path)
        
        # Process PDFs and get texts with sources
        texts, sources = process_pdfs(extract_folder_path)
        if not texts:
            raise ValueError("No valid PDF content found")
        
        # Create documents with source tracking
        documents = split_texts_with_sources(texts, sources)
        
        # Create Pinecone index
        create_pinecone_index(pc, index_name, dimension=768)
        
        # Create vector store
        vectorstore = create_vectorstore(documents, index_name)
        
        logging.info("RAG setup completed successfully")
        return vectorstore
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
