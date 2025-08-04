# src/core/states/base_state.py
from typing import TypedDict, Annotated, Optional, Any, Dict
from langgraph.graph.message import add_messages, AnyMessage
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

# Base state that all modules inherit from
class ChatbotState(TypedDict):
    """Base state for all chatbot modules. Always contains message history."""
    messages: Annotated[list[AnyMessage], add_messages]

# State metadata for registry
@dataclass
class StateMetadata:
    name: str
    description: str
    version: str
    module_path: str
    state_class: type
    adapter_class: type

class TransformationError(Exception):
    """Custom exception for state transformation errors"""
    pass

class ValidationError(Exception):
    """Custom exception for state validation errors"""
    pass
