
# src/modules/text2sql/state.py
from typing import Optional, Dict, Any, List
from src.core.states.base_state import ChatbotState
from src.core.states.state_registry import register_state
from src.core.states.state_adapter import StateAdapter
from src.core.messages.validators import message_extractor
from src.core.states.transformers import state_transformer
from src.services.app_logger import log
from langchain_core.messages import AIMessage, HumanMessage

class Text2SQLState(ChatbotState):
    """Extended state for Text2SQL operations"""
    user_prompt: str
    sql_query: Optional[str] = None
    query_results: Optional[str] = None
    table_schema: Optional[Dict[str, Any]] = None
    execution_error: Optional[str] = None
    query_metadata: Optional[Dict[str, Any]] = None

class Text2SQLAdapter(StateAdapter[Text2SQLState]):
    """State adapter for Text2SQL module following SOLID principles"""
    
    def __init__(self, module_name: str = "text2sql"):
        super().__init__(module_name)
        self.extractor = message_extractor
        self.transformer = state_transformer
    
    def get_state_class(self) -> type:
        """Return the Text2SQLState class"""
        return Text2SQLState
    
    def transform_to_module_state(self, chatbot_state: ChatbotState) -> Text2SQLState:
        """Transform ChatbotState to Text2SQLState"""
        try:
            messages = chatbot_state["messages"]
            
            # Extract user prompt
            user_prompt = self.extractor.extract_user_prompt(messages)
            
            # Extract existing SQL-related data from messages
            sql_query = self._extract_sql_query(messages)
            query_results = self._extract_query_results(messages)
            execution_error = self._extract_execution_error(messages)
            
            # Create enhanced state
            module_state = Text2SQLState(
                messages=messages,
                user_prompt=user_prompt,
                sql_query=sql_query,
                query_results=query_results,
                execution_error=execution_error,
                query_metadata={
                    "intent": self.extractor.detect_intent(messages),
                    "entities": self.extractor.extract_all_entities(messages),
                    "conversation_context": self.extractor.extract_conversation_context(messages)
                }
            )
            
            self.logger.debug("Transformed to Text2SQL state", 
                            user_prompt=user_prompt[:50],
                            has_sql_query=bool(sql_query))
            
            return module_state
            
        except Exception as e:
            self.logger.error("Failed to transform to Text2SQL state", error=str(e))
            # Fallback to basic state
            return Text2SQLState(
                messages=chatbot_state["messages"],
                user_prompt="Error extracting user prompt"
            )
    
    def transform_to_chatbot_state(self, module_state: Text2SQLState) -> ChatbotState:
        """Transform Text2SQLState back to ChatbotState"""
        try:
            # Always preserve message history
            chatbot_state = ChatbotState(messages=module_state["messages"])
            
            # Add any new response messages if results were generated
            if module_state.get("query_results") and not self._results_already_in_messages(module_state):
                response_content = self._format_sql_response(module_state)
                response_message = AIMessage(content=response_content)
                chatbot_state["messages"] = chatbot_state["messages"] + [response_message]
                
                self.logger.debug("Added SQL results to chatbot state")
            
            return chatbot_state
            
        except Exception as e:
            self.logger.error("Failed to transform to chatbot state", error=str(e))
            # Fallback to preserve messages
            return ChatbotState(messages=module_state.get("messages", []))
    
    def validate_state(self, state: Text2SQLState) -> bool:
        """Validate Text2SQLState"""
        try:
            # Check required fields
            if not isinstance(state.get("messages"), list):
                return False
            
            if not isinstance(state.get("user_prompt"), str):
                return False
            
            # Check SQL query format if present
            if state.get("sql_query"):
                sql_query = state["sql_query"].strip().upper()
                # Basic SQL validation
                if not sql_query.startswith(('SELECT', 'WITH')):
                    self.logger.warning("SQL query doesn't start with SELECT or WITH")
                    return False
                
                # Check for dangerous operations
                dangerous_ops = ['DELETE', 'DROP', 'INSERT', 'UPDATE', 'TRUNCATE', 'ALTER']
                if any(op in sql_query for op in dangerous_ops):
                    self.logger.error("Dangerous SQL operation detected")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("State validation failed", error=str(e))
            return False
    
    def _extract_sql_query(self, messages: List) -> Optional[str]:
        """Extract SQL query from message history"""
        for msg in reversed(messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict) and tool_call.get("name") == "db_query_tool":
                        args = tool_call.get("args", {})
                        return args.get("query") or args.get("__arg1")
        return None
    
    def _extract_query_results(self, messages: List) -> Optional[str]:
        """Extract query results from tool messages"""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    if not content.startswith("Error:") and len(content) > 0:
                        return content
        return None
    
    def _extract_execution_error(self, messages: List) -> Optional[str]:
        """Extract SQL execution errors from messages"""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                if isinstance(msg.content, str) and msg.content.startswith("Error:"):
                    return msg.content
        return None
    
    def _results_already_in_messages(self, state: Text2SQLState) -> bool:
        """Check if results are already included in message history"""
        if not state.get("query_results"):
            return True
        
        results = state["query_results"]
        messages = state["messages"]
        
        for msg in reversed(messages):
            if hasattr(msg, 'content') and results in str(msg.content):
                return True
        
        return False
    
    def _format_sql_response(self, state: Text2SQLState) -> str:
        """Format SQL response for user"""
        response_parts = []
        
        response_parts.append("✅ **SQL Query Executed Successfully**")
        response_parts.append("")
        
        if state.get("user_prompt"):
            response_parts.append(f"📋 **Your Question:** {state['user_prompt']}")
            response_parts.append("")
        
        if state.get("sql_query"):
            response_parts.append("🔍 **Generated SQL:**")
            response_parts.append("```sql")
            response_parts.append(state["sql_query"])
            response_parts.append("```")
            response_parts.append("")
        
        if state.get("query_results"):
            response_parts.append("📊 **Results:**")
            response_parts.append("```")
            response_parts.append(state["query_results"])
            response_parts.append("```")
        
        return "\n".join(response_parts)

# Register the Text2SQL state and adapter
@register_state(
    name="text2sql",
    description="State management for Text-to-SQL operations with database querying",
    version="1.0.0"
)
class RegisteredText2SQLAdapter(Text2SQLAdapter):
    """Registered Text2SQL adapter for automatic discovery"""
    
    @classmethod
    def get_state_class(cls) -> type:
        return Text2SQLState