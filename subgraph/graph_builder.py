from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import Any, Literal, TypedDict, cast, List

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import logging
from utils.utils import config

load_dotenv()

logger = logging.getLogger(__name__)


# Custom EnsembleRetriever implementation
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
                docs = retriever._get_relevant_documents(query, run_manager=run_manager)
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


# Custom ContextualCompressionRetriever implementation
class ContextualCompressionRetriever(BaseRetriever):
    """
    Retriever that compresses/reranks documents using a compressor (like Cohere Rerank).
    
    Args:
        base_retriever: The retriever to get initial documents from
        base_compressor: The compressor/reranker to use (e.g., CohereRerank)
    """
    
    base_retriever: BaseRetriever
    base_compressor: Any
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents from base retriever and compress/rerank them."""
        # Get documents from base retriever
        docs = self.base_retriever._get_relevant_documents(query, run_manager=run_manager)
        
        if not docs:
            return []
        
        # Compress/rerank documents using the compressor
        try:
            compressed_docs = self.base_compressor.compress_documents(docs, query)
            return compressed_docs
        except Exception as e:
            logger.warning(f"Error compressing documents: {e}. Returning original docs.")
            return docs


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

def _build_retrievers(documents: list[Document], vectorstore: Chroma) -> ContextualCompressionRetriever:
    
    retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
    retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )
    
    compressor = CohereRerank(top_n=TOP_K_COMPRESSION, model=COHERE_RERANK_MODEL)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    
    return compression_retriever


# Lazy initialization of retriever components
_vectorstore = None
_documents = None
_compression_retriever = None

def get_compression_retriever() -> ContextualCompressionRetriever:
    """Lazy initialization of the compression retriever."""
    global _vectorstore, _documents, _compression_retriever
    
    if _compression_retriever is None:
        _vectorstore = _setup_vectorstore()
        _documents = _load_documents(_vectorstore)
        _compression_retriever = _build_retrievers(_documents, _vectorstore)
    
    return _compression_retriever


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    
    class Response(TypedDict):
        queries: list[str]
    
    logger.info("---Generate Queries---")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)
    logger.info(f"Queries: {queries}")
    return {"queries": response["queries"]}

async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    
    logger.info("---Retrieving documents---")
    logger.info(f"Query for the retrieval process: {state.query}")

    compression_retriever = get_compression_retriever()
    response = compression_retriever.invoke(state.query)
    return {"documents": response}

def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query)) for query in state.queries
    ]
    
builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel, 
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()