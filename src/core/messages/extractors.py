# src/core/messages/extractors.py
from typing import Optional, Dict, Any, List, Union
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from src.services.app_logger import log
import re
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ExtractedData:
    """Container for extracted message data"""
    user_prompt: str
    customer_id: Optional[str] = None
    order_id: Optional[str] = None
    product_id: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

class MessageExtractor:
    """Advanced message parsing and data extraction utilities"""
    
    def __init__(self):
        self.logger = log.get(module="message_extractor")
        self.entity_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for entity extraction"""
        return {
            "customer_id": r'(?:customer|müşteri|cari)(?:\s+(?:id|numarası|no|kodu))?\s*:?\s*([A-Z0-9-]{3,20})',
            "order_id": r'(?:order|sipariş|adisyon)(?:\s+(?:id|numarası|no|kodu))?\s*:?\s*([A-Z0-9-]{3,20})',
            "product_id": r'(?:product|ürün)(?:\s+(?:id|numarası|no|kodu))?\s*:?\s*([A-Z0-9-]{3,20})',
            "invoice_id": r'(?:invoice|fatura)(?:\s+(?:id|numarası|no|kodu))?\s*:?\s*([A-Z0-9-]{3,20})',
            "phone": r'(?:\+90|0)?(?:\s*\(?0?\d{3}\)?\s*\d{3}\s*\d{2}\s*\d{2})',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "amount": r'(?:₺|TL|lira)?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*(?:₺|TL|lira)?',
            "date": r'(\d{1,2}[/.]\d{1,2}[/.]\d{2,4})',
        }
    
    def extract_comprehensive_data(self, messages: List) -> ExtractedData:
        """Extract all available data from message history"""
        try:
            user_prompt = self.extract_user_prompt(messages)
            
            extracted = ExtractedData(
                user_prompt=user_prompt,
                customer_id=self.extract_entity(messages, "customer_id"),
                order_id=self.extract_entity(messages, "order_id"),
                product_id=self.extract_entity(messages, "product_id"),
                intent=self.detect_intent(messages),
                entities=self.extract_all_entities(messages),
                metadata={
                    "message_count": len(messages),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "has_tool_calls": self._has_tool_calls(messages)
                }
            )
            
            self.logger.debug("Comprehensive data extracted", 
                            user_prompt=user_prompt[:50],
                            entities_found=len(extracted.entities),
                            intent=extracted.intent)
            
            return extracted
            
        except Exception as e:
            self.logger.error("Failed to extract comprehensive data", error=str(e))
            return ExtractedData(user_prompt="Error extracting data")
    
    def extract_user_prompt(self, messages: List) -> str:
        """Extract the most recent user prompt"""
        try:
            for msg in reversed(messages):
                if self._is_human_message(msg):
                    content = self._extract_content(msg)
                    if content and content.strip():
                        return content.strip()
            return "No user message found"
        except Exception as e:
            self.logger.error("Failed to extract user prompt", error=str(e))
            return "Error extracting user prompt"
    
    def extract_entity(self, messages: List, entity_type: str) -> Optional[str]:
        """Extract specific entity from messages"""
        if entity_type not in self.entity_patterns:
            self.logger.warning("Unknown entity type", entity_type=entity_type)
            return None
        
        pattern = self.entity_patterns[entity_type]
        
        # Search in all message content
        for msg in messages:
            content = self._extract_content(msg)
            if content:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    return matches[0].strip()
        
        return None
    
    def extract_all_entities(self, messages: List) -> Dict[str, List[str]]:
        """Extract all entities from messages"""
        entities = {}
        
        # Combine all message content
        all_content = " ".join([
            self._extract_content(msg) for msg in messages 
            if self._extract_content(msg)
        ])
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            if matches:
                # Remove duplicates while preserving order
                unique_matches = list(dict.fromkeys(matches))
                entities[entity_type] = [match.strip() for match in unique_matches]
        
        return entities
    
    def detect_intent(self, messages: List) -> str:
        """Advanced intent detection with confidence scoring"""
        user_prompt = self.extract_user_prompt(messages).lower()
        
        intent_patterns = {
            "sql_query": {
                "patterns": [r"sql", r"query", r"database", r"tablo", r"sorgu", r"select", r"rapor"],
                "weight": 1.0
            },
            "customer_service": {
                "patterns": [r"sipariş", r"order", r"müşteri", r"customer", r"problem", r"şikayet", r"destek"],
                "weight": 1.0
            },
            "product_inquiry": {
                "patterns": [r"ürün", r"product", r"fiyat", r"price", r"stok", r"stock", r"katalog"],
                "weight": 0.9
            },
            "payment_support": {
                "patterns": [r"ödeme", r"payment", r"fatura", r"invoice", r"para", r"money", r"tahsilat"],
                "weight": 0.9
            },
            "order_status": {
                "patterns": [r"sipariş durumu", r"order status", r"teslimat", r"delivery", r"kargo"],
                "weight": 0.8
            },
            "general": {
                "patterns": [],
                "weight": 0.1
            }
        }
        
        intent_scores = {}
        
        for intent, config in intent_patterns.items():
            if intent == "general":
                intent_scores[intent] = config["weight"]
                continue
                
            score = 0
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, user_prompt))
                score += matches * config["weight"]
            
            intent_scores[intent] = score
        
        # Return intent with highest score
        detected_intent = max(intent_scores, key=intent_scores.get)
        
        self.logger.debug("Intent detected", 
                        intent=detected_intent, 
                        scores=intent_scores,
                        user_prompt=user_prompt[:50])
        
        return detected_intent
    
    def extract_conversation_context(self, messages: List) -> Dict[str, Any]:
        """Extract conversation context and history"""
        context = {
            "turn_count": 0,
            "user_messages": 0,
            "ai_messages": 0,
            "tool_messages": 0,
            "topics_mentioned": [],
            "last_tool_used": None,
            "conversation_length": len(messages)
        }
        
        for msg in messages:
            if self._is_human_message(msg):
                context["user_messages"] += 1
                context["turn_count"] += 1
            elif self._is_ai_message(msg):
                context["ai_messages"] += 1
                # Check for tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    context["last_tool_used"] = msg.tool_calls[-1].get("name", "unknown")
            elif self._is_tool_message(msg):
                context["tool_messages"] += 1
        
        return context
    
    def _extract_content(self, message) -> str:
        """Extract content from various message formats"""
        if not hasattr(message, 'content'):
            return ""
        
        content = message.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        return ""
    
    def _is_human_message(self, message) -> bool:
        """Check if message is from human"""
        return hasattr(message, 'type') and message.type == 'human'
    
    def _is_ai_message(self, message) -> bool:
        """Check if message is from AI"""
        return hasattr(message, 'type') and message.type == 'ai'
    
    def _is_tool_message(self, message) -> bool:
        """Check if message is from tool"""
        return hasattr(message, 'type') and message.type == 'tool'
    
    def _has_tool_calls(self, messages: List) -> bool:
        """Check if any message contains tool calls"""
        return any(
            hasattr(msg, 'tool_calls') and msg.tool_calls 
            for msg in messages
        )
