
# src/core/states/state_adapter.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, Optional
import asyncio
from src.core.states.base_state import ChatbotState
from src.services.app_logger import log
from langchain_core.messages import AnyMessage

StateType = TypeVar('StateType', bound=ChatbotState)

class StateAdapter(ABC, Generic[StateType]):
    """Abstract base class for all state adapters following SOLID principles."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = log.get(module="state_adapter", adapter=module_name)
    
    @abstractmethod
    def get_state_class(self) -> type:
        """Return the state class this adapter handles"""
        pass
    
    @abstractmethod
    def transform_to_module_state(self, chatbot_state: ChatbotState) -> StateType:
        """Transform base ChatbotState to module-specific state"""
        pass
    
    @abstractmethod
    def transform_to_chatbot_state(self, module_state: StateType) -> ChatbotState:
        """Transform module-specific state back to ChatbotState"""
        pass
    
    @abstractmethod
    def validate_state(self, state: StateType) -> bool:
        """Validate module-specific state"""
        pass
    
    def extract_user_prompt(self, messages: list[AnyMessage]) -> str:
        """Extract the last user prompt from messages - common utility"""
        try:
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    if hasattr(msg, 'content'):
                        if isinstance(msg.content, list) and len(msg.content) > 0:
                            if isinstance(msg.content[0], dict) and 'text' in msg.content[0]:
                                return msg.content[0]['text']
                        elif isinstance(msg.content, str):
                            return msg.content
            return "No user message found"
        except Exception as e:
            self.logger.error("Failed to extract user prompt", error=str(e))
            return "Error extracting user prompt"
    
    async def transform_async(self, chatbot_state: ChatbotState) -> StateType:
        """Async version of transformation for heavy operations"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transform_to_module_state, chatbot_state)
