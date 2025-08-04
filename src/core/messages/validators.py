
# src/core/messages/validators.py
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from src.core.messages.extractors import MessageExtractor
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