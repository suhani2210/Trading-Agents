from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # This stores the history of messages between agents
    messages: Annotated[List[BaseMessage], operator.add]
    ticker: str
    decision: str  # Final: Buy, Hold, Sell
    analysis_report: str