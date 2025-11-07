from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.documents import Document
from utils.utils import reduce_docs

@dataclass(kw_only=True)
class QueryState:
    query: str
    
    
@dataclass(kw_only=True)
class ResearcherState:
    question: str
    queries: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)