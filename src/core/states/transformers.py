
# src/core/states/transformers.py
from typing import Dict, Any, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.core.states.base_state import ChatbotState, TransformationError
from src.services.app_logger import log
import json
import re

class StateTransformer:
    """Utility class for common state transformations"""
    
    def __init__(self):
        self.logger = log.get(module="state_transformer")
    
    def extract_user_prompt(self, messages: list) -> str:
        """Extract the most recent user prompt from messages"""
        try:
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    content = self._extract_message_content(msg)
                    if content:
                        return content
            return "No user message found"
        except Exception as e:
            self.logger.error("Failed to extract user prompt", error=str(e))
            raise TransformationError(f"Could not extract user prompt: {e}")
    
    def _extract_message_content(self, message) -> str:
        """Extract content from various message formats"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and 'text' in content[0]:
                    return content[0]['text']
        return ""
    
    def extract_structured_data(self, messages: list, pattern: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from messages using regex patterns"""
        try:
            user_prompt = self.extract_user_prompt(messages)
            matches = re.findall(pattern, user_prompt, re.IGNORECASE)
            if matches:
                return {"matches": matches, "original_text": user_prompt}
            return None
        except Exception as e:
            self.logger.error("Failed to extract structured data", 
                            pattern=pattern, error=str(e))
            return None
    
    def extract_customer_id(self, messages: list) -> Optional[str]:
        """Extract customer ID from messages"""
        pattern = r'(?:customer|müşteri)(?:\s+(?:id|numarası|no))?\s*:?\s*([A-Z0-9-]+)'
        result = self.extract_structured_data(messages, pattern)
        if result and result["matches"]:
            return result["matches"][0]
        return None
    
    def extract_order_id(self, messages: list) -> Optional[str]:
        """Extract order ID from messages"""
        pattern = r'(?:order|sipariş)(?:\s+(?:id|numarası|no))?\s*:?\s*([A-Z0-9-]+)'
        result = self.extract_structured_data(messages, pattern)
        if result and result["matches"]:
            return result["matches"][0]
        return None
    
    def detect_intent(self, messages: list) -> str:
        """Simple intent detection from user messages"""
        user_prompt = self.extract_user_prompt(messages).lower()
        
        # Define intent patterns
        intent_patterns = {
            "sql_query": [r"sql", r"query", r"database", r"tablo", r"sorgu"],
            "customer_service": [r"sipariş", r"order", r"müşteri", r"customer", r"problem", r"şikayet"],
            "product_info": [r"ürün", r"product", r"fiyat", r"price", r"stok", r"stock"],
            "payment": [r"ödeme", r"payment", r"fatura", r"invoice", r"para", r"money"],
            "general": []  # fallback
        }
        
        for intent, patterns in intent_patterns.items():
            if intent == "general":
                continue
            if any(re.search(pattern, user_prompt) for pattern in patterns):
                self.logger.debug("Intent detected", intent=intent, user_prompt=user_prompt[:50])
                return intent
        
        return "general"
    
    def create_response_message(self, content: str, message_type: str = "ai") -> Union[AIMessage, HumanMessage]:
        """Create a properly formatted response message"""
        if message_type.lower() == "ai":
            return AIMessage(content=content)
        elif message_type.lower() == "human":
            return HumanMessage(content=content)
        else:
            raise TransformationError(f"Unsupported message type: {message_type}")
    
    def merge_states(self, base_state: ChatbotState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge state updates while preserving message history"""
        try:
            result = dict(base_state)
            
            # Always preserve messages
            if "messages" in updates:
                # Merge messages instead of replacing
                existing_messages = result.get("messages", [])
                new_messages = updates["messages"]
                result["messages"] = existing_messages + new_messages
            
            # Add other fields
            for key, value in updates.items():
                if key != "messages":
                    result[key] = value
            
            return result
        except Exception as e:
            self.logger.error("Failed to merge states", error=str(e))
            raise TransformationError(f"Could not merge states: {e}")
    
    def validate_message_history(self, messages: list) -> bool:
        """Validate that message history is properly formatted"""
        try:
            if not isinstance(messages, list):
                return False
            
            for msg in messages:
                if not hasattr(msg, 'type') or not hasattr(msg, 'content'):
                    return False
                
                if msg.type not in ['human', 'ai', 'tool', 'system']:
                    return False
            
            return True
        except Exception:
            return False

# Global transformer instance
state_transformer = StateTransformer()