from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from main_graph.graph_states import AgentState, Router, GradeHallucinations, InputState
from utils.prompt import ROUTER_SYSTEM_PROMPT, RESEARCH_PLAN_SYSTEM_PROMPT, MORE_INFO_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, CHECK_HALLUCINATIONS, RESPONSE_SYSTEM_PROMPT
from subgraph.graph_builder import researcher_graph
from langchain_core.documents import Document
from typing import Any, Literal, Optional, Union
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import logging
from utils.utils import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)  
logging.getLogger("urllib3").setLevel(logging.WARNING) 

logging.getLogger("openai").propagate = False
logging.getLogger("urllib3").propagate = False
logging.getLogger("httpx").propagate = False

GPT_5_MINI = config["llm"]["gpt_5_mini"]
TEMPERATURE = config["llm"]["temperature"]

async def analyze_and_router_query(state: AgentState, *, config: RunnableConfig) -> dict[str, Router]:
    model = ChatOpenAI(model=GPT_5_MINI, temperature=TEMPERATURE, streaming=True)
    messages=[
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---ANALYZE AND ROUTE QUERY---")
    logging.info(f"Messages: {state.messages}")
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}

def route_query(state: AgentState) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    _type = state.router["type"]
    if _type == "environmental":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")

async def create_research_plan(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str] | str]:
    class Plan(TypedDict):
        steps: list[str]
    
    model = ChatOpenAI(model=GPT_5_MINI, temperature=TEMPERATURE,streaming=True)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---Plan Generation---")
    response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}

async def ask_for_more_info(state: AgentState, *, config: RunnableConfig)-> dict[str, list[BaseMessage]]:
    model = ChatOpenAI(model=GPT_5_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def conduct_research(state: AgentState) -> dict[str, Any]:
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    docs = result["documents"]
    step = state.steps[0]
    logging.info(f"\n{len(docs)} documents retrieved in total for the step: {step}.")
    return {"documents": result["documents"], "steps": state.steps[1:]}