from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import Send


from langchain_core.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import logging
from utils.utils import config

load_dotenv()

logger = logging.getLogger(__name__)

VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]

def _setup_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=VECTORSTORE_COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIRECTORY
    )
    
def _load_documents(vectorstore: Chroma) -> list[Document]:
    all_data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []
    
    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise ValueError(f"Expected metadata to be a dict, but got {type(meta)}")
        
        documents.append(Document(page_content=content, metadata=meta))
    return documents