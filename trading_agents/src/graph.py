import os
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from src.tools import get_stock_data

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

# 1. Define what the agents "remember"
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    ticker: str
    data: dict

# 2. Node: The Technical Analyst
def analyst_node(state: AgentState):
    data = get_stock_data(state['ticker'])
    prompt = f"Analyze this data for {state['ticker']}: {data}. Provide a technical sentiment."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "data": data}

# 3. Node: The Researcher (The "Contrarian" Variation)
def researcher_node(state: AgentState):
    # This agent is forced to find risks
    prompt = f"Based on the analysis so far, what are the hidden RISKS for {state['ticker']}? Be critical."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

# 4. Node: Portfolio Manager (The Decider)
def manager_node(state: AgentState):
    prompt = "Review the technical analysis and the risks. Give a final 'BUY', 'SELL', or 'HOLD' decision with a 1-sentence justification."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

# 5. Build the Workflow
workflow = StateGraph(AgentState)
workflow.add_node("analyst", analyst_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("manager", manager_node)

workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "researcher")
workflow.add_edge("researcher", "manager")
workflow.add_edge("manager", END)

app_graph = workflow.compile()