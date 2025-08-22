# src/graphs/core_text2sql_graph.py
"""
Core Text2SQL Graph - Pure SQL Generation and Execution
Bu graph sadece SQL generation, validation, execution ve fixing yapar
Table preparation ve data exploration aşamaları skip edilir
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.graphs.text2sql_graph import (
    SQLGeneratorAgent, SQLValidatorAgent, SQLExecutorAgent, 
    FixSQLAgent, SQLExecutionRouter, QueryParameterAgent
)
from src.models.models import LLM, State
from src.services.app_logger import log
from src.tools.langgraph_sql_tools import LangGraphSQLTools
from src.tools.custom_sql_tools import CustomSQLTools
from langchain_community.utilities import SQLDatabase
from src.services.config_loader import ConfigLoader


@register_graph("core_text2sql")
class CoreText2SQLGraph(BaseGraph):
    """Core Text2SQL Graph - Starts from SQL generation with prepared state"""
    
    def __init__(self, llm: LLM, db: SQLDatabase = None, memory: MemorySaver = None):
        super().__init__(llm=llm, state_class=State, memory=memory)
        self.logger = log.get(module="core_text2sql_graph")
        
        # Database setup
        if db is None:
            config = ConfigLoader.load_config("config/text2sql_config.yaml")
            db = SQLDatabase.from_uri(config.database.uri)
        
        self.db = db
        
        # Initialize tools
        self._setup_tools()
        
        # Initialize agents
        self._create_agents()
        
        self.logger.info("Core Text2SQL Graph initialized")
    
    def _setup_tools(self):
        """Setup SQL tools"""
        langgraph_tools = LangGraphSQLTools(self.db, self.llm.get_chat())
        custom_tools = CustomSQLTools(self.db)
        self.tools = langgraph_tools.get_basic_tools() + custom_tools.get_custom_tools()
        self.logger.info("SQL tools loaded", count=len(self.tools))
    
    def _create_agents(self):
        """Create SQL agents (reuse from text2sql_graph)"""
        self.agents = {
            "query_parameters_agent": QueryParameterAgent(self.llm.get_chat()),
            "sql_generator": SQLGeneratorAgent(self.llm.get_chat()),
            "sql_validator": SQLValidatorAgent(),
            "sql_executor": SQLExecutorAgent(self.tools),
            "fix_sql_agent": FixSQLAgent(self.llm.get_chat()),
        }
        
        # Create router
        self.sql_router = SQLExecutionRouter()
        
        self.logger.info("Core SQL agents created", agents=list(self.agents.keys()))
    
    def build_graph(self):
        """Build core text2sql graph starting from SQL generation"""
        self.logger.info("Building core text2sql graph")
        
        try:
            # Create memory if not provided
            if self.memory is None:
                self.memory = MemorySaver()
            
            # Create state graph
            graph_builder = StateGraph(State)
            
            # Add agents as nodes (reuse existing agents)
            for agent_name, agent_instance in self.agents.items():
                graph_builder.add_node(agent_name, agent_instance)
            
            # Define core SQL workflow
            # START -> query_parameters_agent -> sql_generator -> sql_validator -> sql_executor -> [fix_sql_agent] -> END
            graph_builder.add_edge(START, "query_parameters_agent")
            graph_builder.add_edge("query_parameters_agent", "sql_generator")
            graph_builder.add_edge("sql_generator", "sql_validator") 
            graph_builder.add_edge("sql_validator", "sql_executor")
            
            # Conditional routing after SQL execution
            graph_builder.add_conditional_edges(
                "sql_executor",
                self.sql_router.route,
                {
                    "fix_sql": "fix_sql_agent",  # Router returns "fix_sql", maps to "fix_sql_agent" node
                    "end": END
                }
            )
            
            # Fix SQL agent goes to END
            graph_builder.add_edge("fix_sql_agent", END)
            
            # Compile graph
            compiled_graph = graph_builder.compile(
                checkpointer=self.memory,
                name="core_text2sql_graph"
            )
            
            self.logger.info("Core text2sql graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build core text2sql graph", error=str(e))
            raise
    
    def invoke(self, state: State, config=None) -> State:
        """Invoke core text2sql workflow with prepared state"""
        try:
            # Validate that required state exists
            if not state.get("relevant_tables"):
                self.logger.error("No relevant_tables found in state")
                state["is_error"] = True
                state["error_message"] = "Required table information missing"
                return state
            
            # Check for messages instead of user_query
            messages = state.get("messages", [])
            if not messages:
                self.logger.error("No messages found in state") 
                state["is_error"] = True
                state["error_message"] = "Messages missing"
                return state
            
            # Convert messages to user_query for core agents
            conversation_context = "\n".join([
                msg.content if hasattr(msg, 'content') else str(msg) 
                for msg in messages if msg
            ])
            state["user_query"] = conversation_context
            
            # Build graph if needed
            if not hasattr(self, '_compiled_graph'):
                self._compiled_graph = self.build_graph()
            
            # Execute core SQL workflow
            result_state = self._compiled_graph.invoke(state, config)
            
            self.logger.info("Core Text2SQL workflow completed successfully")
            return result_state
            
        except Exception as e:
            self.logger.error("Core Text2SQL workflow failed", error=str(e))
            state["is_error"] = True
            state["error_message"] = f"Core Text2SQL execution failed: {str(e)}"
            return state


# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    """Example usage of Core Text2SQL Graph"""
    
    from src.models.models import LLM
    from src.graphs.text2sql_graph import Table
    
    # Initialize LLM
    llm = LLM(model="gpt-4o-mini", temperature=0.0)
    
    # Create core graph
    core_graph = CoreText2SQLGraph(llm)
    
    # Example prepared state (would come from generic_sql_graph)
    prepared_state = {
        "user_query": "Show me all products with their prices",
        "relevant_tables": [
            Table(
                name="Products",
                columns=["[ProductID] (int): Product ID", "[ProductName] (nvarchar): Product Name"],
                relations=[],
                schema="-- Table: Products",
                distinct_values={"ProductID": ["1", "2", "3"], "ProductName": ["Product A", "Product B"]}
            )
        ],
        "table_schemas": "-- Table: Products",
        "all_tables": ["Products"],
        "is_error": False,
        "error_message": ""
    }
    
    # Run core workflow
    result = core_graph.invoke(prepared_state)
    
    print("🚀 Core Text2SQL Result:")
    print(f"SQL Generated: {result.get('generated_sql', 'N/A')}")
    print(f"SQL Result: {result.get('sql_result', 'N/A')}")
    print(f"Is Error: {result.get('is_error', False)}")