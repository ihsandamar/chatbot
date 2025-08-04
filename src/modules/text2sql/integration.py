
# src/modules/text2sql/integration.py
from typing import Dict, Any
from langchain_community.utilities import SQLDatabase
from src.graphs.text2sql_graph import Text2SQLGraph
from src.modules.text2sql.state import Text2SQLAdapter, Text2SQLState
from src.core.states.base_state import ChatbotState
from src.services.app_logger import log

class Text2SQLModule:
    """Integration layer for Text2SQL module with state management"""
    
    def __init__(self, llm, db_uri: str):
        self.llm = llm
        self.db = SQLDatabase.from_uri(db_uri)
        self.graph = Text2SQLGraph(llm=llm, db=self.db)
        self.adapter = Text2SQLAdapter()
        self.logger = log.get(module="text2sql_integration")
    
    def process_request(self, chatbot_state: ChatbotState) -> ChatbotState:
        """Process Text2SQL request with state transformation"""
        try:
            # Transform to module-specific state
            text2sql_state = self.adapter.transform_to_module_state(chatbot_state)
            
            # Validate state
            if not self.adapter.validate_state(text2sql_state):
                self.logger.error("Invalid Text2SQL state")
                return self._create_error_response(chatbot_state, "Invalid request format")
            
            # Execute the graph
            graph_instance = self.graph.build_graph()
            result = graph_instance.invoke(
                {"messages": text2sql_state["messages"]},
                config={"configurable": {"thread_id": "text2sql_session"}}
            )
            
            # Update state with results
            updated_state = text2sql_state.copy()
            updated_state["messages"] = result["messages"]
            
            # Transform back to chatbot state
            final_state = self.adapter.transform_to_chatbot_state(updated_state)
            
            self.logger.info("Text2SQL request processed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error("Text2SQL processing failed", error=str(e))
            return self._create_error_response(chatbot_state, f"Processing error: {str(e)}")
    
    def _create_error_response(self, original_state: ChatbotState, error_message: str) -> ChatbotState:
        """Create error response while preserving message history"""
        from langchain_core.messages import AIMessage
        
        error_response = AIMessage(
            content=f"❌ **Text2SQL Error**\n\n{error_message}\n\nPlease try rephrasing your question or contact support."
        )
        
        return ChatbotState(
            messages=original_state["messages"] + [error_response]
        )
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information and capabilities"""
        return {
            "name": "Text2SQL",
            "version": "1.0.0",
            "description": "Natural language to SQL query conversion and execution",
            "capabilities": [
                "Database schema exploration",
                "SQL query generation from natural language",
                "Query execution and result formatting",
                "Error handling and query correction"
            ],
            "supported_databases": ["MSSQL", "PostgreSQL", "MySQL", "SQLite"],
            "state_fields": [
                "user_prompt",
                "sql_query", 
                "query_results",
                "table_schema",
                "execution_error",
                "query_metadata"
            ]
        }