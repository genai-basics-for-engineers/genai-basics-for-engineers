from __future__ import annotations
import os
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# outputs/ ディレクトリの自動作成
Path("scripts/chapter7/outputs").mkdir(parents=True, exist_ok=True)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    step_count: int
    satisfied: bool
    findings: Annotated[List[str], operator.add]

@tool
def offline_search(query: str) -> dict:
    """サンプル検索ツール"""
    return {"query": query, "results": []}

def agent_node(state: AgentState) -> AgentState:
    return state

def compose_node(state: AgentState) -> AgentState:
    return state

def should_continue(state: AgentState) -> Literal["tools", "compose", END]:
    if state["step_count"] >= 5:
        return END
    if state["satisfied"]:
        return END
    return "tools"

# グラフ構築
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([offline_search]))
graph.add_node("compose", compose_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "compose": "compose",
        END: END,
    },
)
graph.add_edge("tools", "compose")
graph.add_edge("compose", "agent")

app = graph.compile()

# Mermaid可視化
print(app.get_graph().draw_mermaid())
