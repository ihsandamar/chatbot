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

# src/core/messages/validators.py
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from src.services.app_logger import log
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"  
    LENIENT = "lenient"

@dataclass
class ValidationResult:
    """Result of message validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class MessageValidator:
    """Comprehensive message validation utilities"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = log.get(module="message_validator")
    
    def validate_message_history(self, messages: List) -> ValidationResult:
        """Validate entire message history"""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Basic structure validation
            if not isinstance(messages, list):
                errors.append("Messages must be a list")
                return ValidationResult(False, errors, warnings, metadata)
            
            if len(messages) == 0:
                warnings.append("Empty message history")
            
            # Validate each message
            for i, msg in enumerate(messages):
                msg_result = self.validate_single_message(msg, i)
                errors.extend(msg_result.errors)
                warnings.extend(msg_result.warnings)
            
            # Validate conversation flow
            flow_result = self.validate_conversation_flow(messages)
            errors.extend(flow_result.errors)
            warnings.extend(flow_result.warnings)
            
            # Collect metadata
            metadata = {
                "total_messages": len(messages),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "message_types": self._count_message_types(messages)
            }
            
            is_valid = len(errors) == 0
            
            if not is_valid:
                self.logger.warning("Message validation failed", 
                                  errors=errors, 
                                  warnings=warnings)
            
            return ValidationResult(is_valid, errors, warnings, metadata)
            
        except Exception as e:
            self.logger.error("Validation failed with exception", error=str(e))
            return ValidationResult(False, [f"Validation exception: {e}"], [], {})
    
    def validate_single_message(self, message, index: int) -> ValidationResult:
        """Validate a single message"""
        errors = []
        warnings = []
        
        # Check basic message structure
        if not hasattr(message, 'type'):
            errors.append(f"Message {index}: Missing 'type' attribute")
        
        if not hasattr(message, 'content'):
            errors.append(f"Message {index}: Missing 'content' attribute")
        elif message.content is None or (isinstance(message.content, str) and not message.content.strip()):
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Message {index}: Empty content")
            else:
                warnings.append(f"Message {index}: Empty content")
        
        # Validate message type
        if hasattr(message, 'type'):
            valid_types = ['human', 'ai', 'tool', 'system']
            if message.type not in valid_types:
                errors.append(f"Message {index}: Invalid type '{message.type}'")
        
        # Tool message specific validation
        if hasattr(message, 'type') and message.type == 'tool':
            if not hasattr(message, 'tool_call_id'):
                errors.append(f"Message {index}: Tool message missing tool_call_id")
        
        # AI message with tool calls validation
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for j, tool_call in enumerate(message.tool_calls):
                if not isinstance(tool_call, dict):
                    errors.append(f"Message {index}, tool_call {j}: Must be dict")
                elif 'name' not in tool_call:
                    errors.append(f"Message {index}, tool_call {j}: Missing 'name'")
        
        return ValidationResult(len(errors) == 0, errors, warnings, {})
    
    def validate_conversation_flow(self, messages: List) -> ValidationResult:
        """Validate logical conversation flow"""
        errors = []
        warnings = []
        
        if not messages:
            return ValidationResult(True, [], [], {})
        
        # Check for orphaned tool messages
        tool_call_ids = set()
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict) and 'id' in tool_call:
                        tool_call_ids.add(tool_call['id'])
        
        for i, msg in enumerate(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id not in tool_call_ids:
                    warnings.append(f"Message {i}: Orphaned tool message with id {msg.tool_call_id}")
        
        # Check conversation starts with human message (optional)
        if self.validation_level == ValidationLevel.STRICT:
            if messages and not (hasattr(messages[0], 'type') and messages[0].type == 'human'):
                warnings.append("Conversation should typically start with human message")
        
        return ValidationResult(len(errors) == 0, errors, warnings, {})
    
    def validate_state_compatibility(self, state: Dict[str, Any]) -> ValidationResult:
        """Validate if state is compatible with ChatbotState"""
        errors = []
        warnings = []
        
        # Check required fields
        if 'messages' not in state:
            errors.append("State missing required 'messages' field")
        else:
            # Validate messages field
            msg_result = self.validate_message_history(state['messages'])
            errors.extend([f"Messages field: {error}" for error in msg_result.errors])
            warnings.extend([f"Messages field: {warning}" for warning in msg_result.warnings])
        
        # Check for unknown fields in strict mode
        if self.validation_level == ValidationLevel.STRICT:
            known_base_fields = {'messages'}
            unknown_fields = set(state.keys()) - known_base_fields
            if unknown_fields:
                warnings.append(f"Unknown fields in base state: {unknown_fields}")
        
        metadata = {
            "field_count": len(state.keys()),
            "has_messages": 'messages' in state,
            "message_count": len(state.get('messages', []))
        }
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def _count_message_types(self, messages: List) -> Dict[str, int]:
        """Count messages by type"""
        counts = {"human": 0, "ai": 0, "tool": 0, "system": 0, "unknown": 0}
        
        for msg in messages:
            msg_type = getattr(msg, 'type', 'unknown')
            if msg_type in counts:
                counts[msg_type] += 1
            else:
                counts['unknown'] += 1
        
        return counts
    
    def suggest_fixes(self, validation_result: ValidationResult) -> List[str]:
        """Suggest fixes for validation errors"""
        suggestions = []
        
        for error in validation_result.errors:
            if "Missing 'type' attribute" in error:
                suggestions.append("Add 'type' attribute to message (human/ai/tool/system)")
            elif "Missing 'content' attribute" in error:
                suggestions.append("Add 'content' attribute with message text")
            elif "Empty content" in error:
                suggestions.append("Provide non-empty content for the message")
            elif "Invalid type" in error:
                suggestions.append("Use valid message type: human, ai, tool, or system")
            elif "Tool message missing tool_call_id" in error:
                suggestions.append("Add tool_call_id to tool messages")
            elif "State missing required 'messages' field" in error:
                suggestions.append("Add 'messages' field to state containing list of messages")
        
        return suggestions

# Global instances
message_extractor = MessageExtractor()
message_validator = MessageValidator()