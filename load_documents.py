"""
Script to load PDF documents into the vector database.
Run this before using the application to populate the retriever.
"""

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.utils import config
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents_to_vectorstore():
    """Load PDF documents into Chroma vector database."""
    
    # Get config
    filepath = config["retriever"]["file"]
    headers_to_split_on = config["retriever"]["headers_to_split_on"]
    collection_name = config["retriever"]["collection_name"]
    persist_directory = config["retriever"]["directory"]
    
    logger.info(f"Loading documents from: {filepath}")
    
    # Convert PDF to markdown
    try:
        converter = DocumentConverter()
        result = converter.convert(filepath)
        markdown_document = result.document.export_to_markdown()
        logger.info("PDF converted to markdown successfully")
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return False
    
    # Split into chunks
    try:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        docs_list = markdown_splitter.split_text(markdown_document)
        logger.info(f"Document split into {len(docs_list)} chunks")
    except Exception as e:
        logger.error(f"Error splitting document: {e}")
        return False
    
    # Create embeddings and store in Chroma
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=docs_list,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        logger.info(f"✓ Successfully loaded {len(docs_list)} documents into vector database")
        logger.info(f"✓ Collection: {collection_name}")
        logger.info(f"✓ Directory: {persist_directory}")
        return True
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Loading Documents into Vector Database")
    print("="*60 + "\n")
    
    success = load_documents_to_vectorstore()
    
    if success:
        print("\n" + "="*60)
        print("  ✓ Documents loaded successfully!")
        print("  You can now run the application.")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("  ✗ Failed to load documents")
        print("  Please check the error messages above")
        print("="*60 + "\n")
