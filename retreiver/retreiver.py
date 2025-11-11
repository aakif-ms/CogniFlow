from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from utils.utils import config

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from typing import List, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines multiple retrievers with weighted scores.
    
    This retriever takes multiple retrievers and combines their results using
    weighted scoring based on document rank positions.
    
    Args:
        retrievers: List of retriever instances to combine
        weights: List of weights for each retriever (should sum to 1.0 or be normalized)
    """
    
    retrievers: List[BaseRetriever]
    weights: List[float]
    c: int = 60  # Constant for reciprocal rank fusion
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float], **kwargs):
        """Initialize the ensemble retriever."""
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        super().__init__(retrievers=retrievers, weights=normalized_weights, **kwargs)
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get relevant documents from all retrievers and combine with weighted RRF.
        
        Uses Reciprocal Rank Fusion (RRF) algorithm with weights to combine results.
        """
        # Get documents from all retrievers
        doc_lists = []
        for retriever in self.retrievers:
            try:
                docs = retriever.get_relevant_documents(query)
                doc_lists.append(docs)
            except Exception as e:
                logger.warning(f"Error retrieving from {retriever.__class__.__name__}: {e}")
                doc_lists.append([])
        
        # Use document content + metadata as key for deduplication
        def doc_key(doc: Document) -> str:
            return f"{doc.page_content}:{str(doc.metadata)}"
        
        # Calculate weighted RRF scores
        doc_scores = {}
        doc_map = {}
        
        for docs, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(docs):
                key = doc_key(doc)
                # RRF score: weight / (k + rank)
                score = weight / (self.c + rank + 1)
                
                if key in doc_scores:
                    doc_scores[key] += score
                else:
                    doc_scores[key] = score
                    doc_map[key] = doc
        
        # Sort by score and return documents
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[key] for key in sorted_keys]

class DocumentProcessor:
    def __init__(self, headers_to_split_on: List[str]):
        self.headers_to_split_on = headers_to_split_on
        
    def process(self, source: Any) -> List[str]:
        try:
            logger.info("Starting document processing")
            converter = DocumentConverter()
            markdown_document = converter.convert(source).document.export_to_markdown()
            markdown_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)
            docs_list = markdown_splitter.split_text(markdown_document)
            logger.info("Document processed successfully")
            return docs_list
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise RuntimeError(f"Error processing document: {e}")

class IndexBuilder:
    def __init__(self, docs_list: List[str],
                 collection_name: str,
                 persist_directory: str,
                 load_documents: bool):
        self.docs_list = docs_list
        self.collection_name = collection_name
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.load_documents = load_documents
        
    def build_vectorstore(self):
        embeddings = OpenAIEmbeddings()
        try:
            logger.info("Building vectorstore")
            self.vectorstore = Chroma.from_documents(
                persist_directory=self.persist_directory,
                documents=self.docs_list,
                collection_name=self.collection_name,
                embedding=embeddings
            )
            logger.info("Vectorstore built successfully")
        except Exception as e:
            logger.error(f"Error building vectorstore: {e}")
            raise RuntimeError(f"Error building vectorstore: {e}")

    def build_retrievers(self):
        try:
            logger.info("Building BM25 retriever.")
            bm25_retriever = BM25Retriever.from_documents(self.docs_list, search_kwargs={"k": 4}) # type: ignore
            logger.info("Building vector-based retrievers")
            retriever_vanilla = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            retriever_mmr = self.vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 4}
            )
            
            logger.info("Combining retrievers into an ensemble retriever")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_mmr, retriever_vanilla, bm25_retriever],
                weights=[0.3, 0.3, 0.4]
            )
            logger.info("Retrievers build successfully")
            return ensemble_retriever
        except Exception as e:
            logger.error(f"Error building retrievers: {e}")
            raise RuntimeError(f"Error building retrievers: {e}")

if __name__ == "__main__":
    headers_to_split_on = config["retriever"]["headers_to_split_on"]
    filepath = config["retriever"]["file"]
    collection_name = config["retriever"]["collection_name"]
    load_documents = config["retriever"]["load_documents"]

    print("Retriever entry")
    if load_documents:
        logger.info("Initializing document processor.")
        processor = DocumentProcessor(headers_to_split_on) 
        try:        
            docs_list = processor.process(filepath)    
            logger.info(f"{len(docs_list)} chunks generated.") 
        except RuntimeError as e:        
            logger.info(f"Failed to process document: {e}")        
            exit(1)

    logger.info("Initializing index builder.")
    index_builder = IndexBuilder(docs_list, collection_name, persist_directory="vector_db", load_documents=load_documents)
    index_builder.build_vectorstore()

    try:
        ensemble_retriever = index_builder.build_retrievers()
        logger.info("Index and retrievers built successfully. Ready for use.")
    except RuntimeError as e:
        logger.critical(f"Failed to build index or retrievers: {e}")
        exit(1)
