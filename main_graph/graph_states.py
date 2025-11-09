from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from utils.utils import reduce_docs
from pydantic import BaseModel, Field

@dataclass(kw_only=True)
class InputState:
    messages: Annotated[list[AnyMessage], add_messages]

class Router(TypedDict):
    logic: str
    type: Literal["more-info", "environmental", "general"]

class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )

@dataclass(kw_only=True)
class AgentState(InputState):
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    steps: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    hallucination: GradeHallucinations = field(default_factory=lambda: GradeHallucinations(binary_score="0"))