"""This package provides the core Agent interface and supporting components for building AI agents with the SDK.

It includes:

- Agent: The main interface for interacting with AI models and tools
- StructuredAgent: Enhanced agent with native Pydantic model support for structured outputs
- ConversationManager: Classes for managing conversation history and context windows
"""

from .agent import Agent
from .agent_result import AgentResult
from .conversation_manager import ConversationManager, NullConversationManager, SlidingWindowConversationManager
from .structured_agent import StructuredAgent, structured_query

__all__ = [
    "Agent",
    "StructuredAgent", 
    "AgentResult",
    "ConversationManager",
    "NullConversationManager",
    "SlidingWindowConversationManager",
    "structured_query",
]
