# src/graphs/text2sql_graph.py - LangGraph Native with Runnable Agents
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Literal, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from typing_extensions import NotRequired

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.tools.langgraph_sql_tools import LangGraphSQLTools
from src.tools.custom_sql_tools import CustomSQLTools
from src.services.app_logger import log

from langchain_community.utilities import SQLDatabase
from src.services.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader.load_config("config/text2sql_config.yaml")

# ================================
# DYNAMIC AI TABLE ANALYZER TOOL
# ================================
@tool
def analyze_user_query_for_tables(user_query: str, all_tables: str) -> str:
    """AI tool to dynamically analyze user query and select most relevant database tables."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    tables_list = [t.strip() for t in all_tables.split(',') if t.strip()]
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=config.llm.api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database expert. Analyze the user query and select the most relevant tables.

INSTRUCTIONS:
- Understand the user's intent and data needs
- Select 3-8 most relevant tables that contain the requested data
- Consider table relationships and dependencies
- Focus on tables that directly relate to the user's question
- Return ONLY table names separated by commas, no explanations

EXAMPLES:
- "Products data" → Tables with "Product" in name
- "Customer information" → Tables with "Customer" related data
- "Sales report" → Tables related to sales, orders, transactions
- "BOM listing" → Tables related to Bill of Materials"""),
            ("human", f"""User Query: {user_query}

Available Tables: {all_tables}

Select the most relevant tables (comma separated):""")
        ])
        
        response = llm.invoke(prompt.format_messages())
        selected = [t.strip() for t in response.content.split(',') if t.strip()]
        valid_selected = [t for t in selected if t in tables_list]
        
        result = ', '.join(valid_selected[:8]) if valid_selected else ', '.join(tables_list[:5])
        print(f"[AI ANALYSIS] Query: {user_query[:50]} → Tables: {result}")
        return result
        
    except Exception as e:
        print(f"[AI ANALYSIS ERROR] {e}")
        return ', '.join(tables_list[:5])

# ================================
# STATE DEFINITION
# ================================
@dataclass
class UserParameters:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    group_by_fields: List[str] = None
    order_by_fields: List[str] = None
    limit: int = 100
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.group_by_fields is None:
            self.group_by_fields = []
        if self.filters is None:
            self.filters = {}

@dataclass
class ErrorInfo:
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    last_failed_query: str = ""

class Text2SQLState(TypedDict):
    messages: List[Any]
    report_module: NotRequired[Literal["masterreport", "accounting", "dynamicreport"]]
    user_parameters: NotRequired[UserParameters]
    all_tables: NotRequired[List[str]]
    relevant_tables: NotRequired[List[str]]
    table_schemas: NotRequired[str]
    generated_sql: NotRequired[str]
    query_result: NotRequired[str]
    error_info: NotRequired[ErrorInfo]
    schema_loaded: NotRequired[bool]
    tables_listed: NotRequired[bool]

# ================================
# UTILITY CLASSES
# ================================
class StateManager:
    @staticmethod
    def init_state(state: Text2SQLState) -> Text2SQLState:
        defaults = {
            "user_parameters": UserParameters(),
            "error_info": ErrorInfo(),
            "schema_loaded": False,
            "tables_listed": False,
            "all_tables": [],
            "relevant_tables": [],
            "table_schemas": ""
        }
        for key, value in defaults.items():
            if key not in state:
                state[key] = value
        return state
    
    @staticmethod
    def get_user_question(messages: List) -> str:
        for msg in reversed(messages):
            if isinstance(msg, str):
                return msg
            elif hasattr(msg, 'type') and msg.type == 'human' and hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    return msg.content
        if messages and isinstance(messages[0], str):
            return messages[0]
        return "Soru bulunamadı"

# ================================
# BASE AGENT - LangGraph Runnable Native
# ================================
class BaseAgent(Runnable, ABC):
    """Base Agent class - LangGraph Runnable Native"""
    
    def __init__(self, name: str, tools: List = None):
        super().__init__()
        self.name = name
        self.tools = tools or []
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.logger = log.get(module="text2sql", agent=name)
    
    @abstractmethod
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        """LangGraph Runnable invoke method"""
        pass
    
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a tool by name with error handling"""
        tool = self.tools_dict.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in agent '{self.name}'")
        
        try:
            result = tool.invoke(args)
            self.logger.info("Tool executed successfully", tool=tool_name, args=str(args)[:100])
            return result
        except Exception as e:
            self.logger.error("Tool execution failed", tool=tool_name, error=str(e))
            raise e

# ================================
# AGENT IMPLEMENTATIONS
# ================================
class InitializationAgent(BaseAgent):
    """State initialization agent"""
    
    def __init__(self):
        super().__init__("InitializationAgent", tools=[])
    
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        self.logger.info("Initializing Text2SQL state")
        state = StateManager.init_state(state)
        user_question = StateManager.get_user_question(state["messages"])
        self.logger.info("User question received", question=user_question[:100])
        return {"messages": state["messages"]}

class TableListingAgent(BaseAgent):
    """Database table listing agent"""
    
    def __init__(self, tools: List):
        # Filter only the sql_db_list_tables tool
        list_tools = [t for t in tools if t.name == "sql_db_list_tables"]
        super().__init__("TableListingAgent", tools=list_tools)
    
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        self.logger.info("Listing database tables")
        messages = state["messages"]
        
        try:
            # Execute list tables tool
            result = self.execute_tool("sql_db_list_tables", {})
            
            # Parse table list
            all_tables = [t.strip().strip("'\"") for t in str(result).replace('[', '').replace(']', '').replace("'", "").split(',')]
            all_tables = [t for t in all_tables if t]
            
            self.logger.info("Tables listed successfully", count=len(all_tables))
            
            return {
                "messages": messages + [AIMessage(content=f"Found {len(all_tables)} tables in database")],
                "all_tables": all_tables,
                "tables_listed": True
            }
        except Exception as e:
            self.logger.error("Failed to list tables", error=str(e))
            return {"messages": messages + [AIMessage(content=f"Error listing tables: {str(e)}")]}

class SchemaAnalysisAgent(BaseAgent):
    """AI-driven schema analysis agent"""
    
    def __init__(self, tools: List):
        # Filter only the AI analysis tool
        analysis_tools = [t for t in tools if t.name == "analyze_user_query_for_tables"]
        super().__init__("SchemaAnalysisAgent", tools=analysis_tools)
    
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        self.logger.info("Analyzing user query for relevant tables")
        messages = state["messages"]
        all_tables = state.get("all_tables", [])
        
        if not all_tables:
            return {"messages": messages + [AIMessage(content="No tables available for analysis")]}
        
        user_question = StateManager.get_user_question(messages)
        
        try:
            # Execute AI analysis tool
            result = self.execute_tool("analyze_user_query_for_tables", {
                "user_query": user_question,
                "all_tables": ", ".join(all_tables)
            })
            
            # Parse selected tables
            selected_tables = [t.strip() for t in str(result).split(',') if t.strip()]
            
            self.logger.info("AI analysis completed", 
                           user_question=user_question[:50],
                           selected_count=len(selected_tables),
                           selected_tables=selected_tables)
            
            return {
                "messages": messages + [AIMessage(content=f"AI selected {len(selected_tables)} relevant tables: {', '.join(selected_tables)}")],
                "relevant_tables": selected_tables
            }
        except Exception as e:
            self.logger.error("AI analysis failed", error=str(e))
            # Fallback: use first 5 tables
            fallback_tables = all_tables[:5]
            return {
                "messages": messages + [AIMessage(content=f"Using fallback tables: {', '.join(fallback_tables)}")],
                "relevant_tables": fallback_tables
            }

class SchemaRetrievalAgent(BaseAgent):
    """Database schema retrieval agent"""
    
    def __init__(self, tools: List):
        # Filter only the schema tool
        schema_tools = [t for t in tools if t.name == "sql_db_schema"]
        super().__init__("SchemaRetrievalAgent", tools=schema_tools)
    
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        self.logger.info("Retrieving schema for selected tables")
        messages = state["messages"]
        relevant_tables = state.get("relevant_tables", [])
        
        if not relevant_tables:
            return {"messages": messages + [AIMessage(content="No relevant tables found for schema retrieval")]}
        
        try:
            # Execute schema tool
            result = self.execute_tool("sql_db_schema", {"table_names": ", ".join(relevant_tables)})
            
            self.logger.info("Schema retrieved successfully", 
                           tables=relevant_tables,
                           schema_length=len(str(result)))
            
            return {
                "messages": messages + [AIMessage(content=f"Schema loaded for {len(relevant_tables)} tables")],
                "table_schemas": str(result),
                "schema_loaded": True
            }
        except Exception as e:
            self.logger.error("Schema retrieval failed", error=str(e))
            return {"messages": messages + [AIMessage(content=f"Error retrieving schema: {str(e)}")]}

class QueryGenerationAgent(BaseAgent):
    """SQL query generation and execution agent"""
    
    def __init__(self, llm, tools: List):
        # Filter only the SQL execution tool
        sql_tools = [t for t in tools if t.name == "db_query_tool"]
        super().__init__("QueryGenerationAgent", tools=sql_tools)
        self.llm = llm
        self._setup_model()
    
    def _setup_model(self):
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MSSQL expert. Generate and execute SQL queries based on the provided schema.

INSTRUCTIONS:
- Use EXACT table and column names from the schema
- Generate simple, working SQL queries first
- Apply user parameters for filtering and limiting
- Focus on the user's specific request

The schema and user parameters are provided in the conversation."""),
            ("placeholder", "{messages}")
        ])
        
        self.query_model = self.query_prompt | self.llm.bind_tools(
            self.tools, tool_choice="required"
        )
    
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        self.logger.info("Generating and executing SQL query")
        messages = state["messages"]
        user_parameters = state.get("user_parameters", UserParameters())
        
        # Convert dict to UserParameters if needed
        if isinstance(user_parameters, dict):
            user_parameters = UserParameters(**user_parameters)
        
        # Check if schema is loaded
        if not state.get("schema_loaded", False):
            return {"messages": messages + [AIMessage(content="Schema not loaded, cannot generate query")]}
        
        # Add parameter context
        param_context = f"""
User Parameters:
- Limit: {user_parameters.limit}
- Group By: {', '.join(user_parameters.group_by_fields) if user_parameters.group_by_fields else 'None'}
- Order By: {', '.join(user_parameters.order_by_fields) if user_parameters.order_by_fields else 'None'}
- Filters: {user_parameters.filters if user_parameters.filters else 'None'}

Please generate and execute a SQL query based on the schema and user request.
"""
        
        enhanced_messages = messages + [HumanMessage(content=param_context)]
        
        try:
            # Generate query with LLM
            response = self.query_model.invoke({"messages": enhanced_messages})
            
            # Execute the generated SQL
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.get("name") == "db_query_tool":
                    query = tool_call["args"].get("query", "") or tool_call["args"].get("__arg1", "")
                    
                    try:
                        # Execute SQL tool
                        result = self.execute_tool("db_query_tool", {"query": query})
                        
                        self.logger.info("SQL executed successfully", 
                                       query=query[:100],
                                       result_length=len(str(result)))
                        
                        return {
                            "messages": messages + [
                                response,
                                AIMessage(content=f"Query executed successfully.")
                            ],
                            "generated_sql": query,
                            "query_result": str(result)
                        }
                    except Exception as e:
                        self.logger.error("SQL execution failed", error=str(e))
                        return {
                            "messages": messages + [
                                response,
                                AIMessage(content=f"SQL Error: {str(e)}")
                            ],
                            "generated_sql": query,
                            "error_info": ErrorInfo(error_message=str(e), last_failed_query=query)
                        }
            
            return {"messages": messages + [response]}
            
        except Exception as e:
            self.logger.error("Query generation failed", error=str(e))
            return {"messages": messages + [AIMessage(content=f"Query generation error: {str(e)}")]}

class FinalAnswerAgent(BaseAgent):
    """Final answer formatting agent"""
    
    def __init__(self):
        super().__init__("FinalAnswerAgent", tools=[])
    
    def invoke(self, state: Text2SQLState, config=None) -> Dict[str, Any]:
        self.logger.info("Creating final answer")
        messages = state["messages"]
        report_module = state.get("report_module", "dynamicreport")
        user_parameters = state.get("user_parameters", UserParameters())
        relevant_tables = state.get("relevant_tables", [])
        generated_sql = state.get("generated_sql", "")
        query_result = state.get("query_result", "")
        
        # Convert dict to UserParameters if needed
        if isinstance(user_parameters, dict):
            user_parameters = UserParameters(**user_parameters)
        
        user_question = StateManager.get_user_question(messages)
        
        if query_result and not query_result.startswith("Error:"):
            self.logger.info("Creating successful result")
            
            final_text = f"""✅ **{report_module.title()} Raporu Başarıyla Oluşturuldu!**

📋 **Sorunuz:** {user_question}

📊 **Rapor Modülü:** {report_module.title()}

📋 **AI Seçimi:** {', '.join(relevant_tables) if relevant_tables else 'Belirlenmedi'}

⚙️ **Parametreler:**
- Limit: {user_parameters.limit}
- Gruplama: {', '.join(user_parameters.group_by_fields) if user_parameters.group_by_fields else 'Yok'}
- Sıralama: {', '.join(user_parameters.order_by_fields) if user_parameters.order_by_fields else 'Varsayılan'}

🔍 **Çalıştırılan SQL:**
```sql
{generated_sql}
```

📊 **Sonuçlar:**
{query_result}

💡 **Özet:** AI-driven analiz ile rapor başarıyla oluşturuldu."""
        else:
            self.logger.error("Creating error result")
            final_text = f"""❌ **Rapor Oluşturma Hatası**

📋 **Sorunuz:** {user_question}
📊 **Rapor Modülü:** {report_module.title()}

Query sonucu bulunamadı veya hata oluştu. Lütfen tekrar deneyin."""
        
        response = AIMessage(content=final_text)
        return {"messages": messages + [response]}

# ================================
# TOOL MANAGER
# ================================
class SQLToolManager:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.logger = log.get(module="text2sql", component="tool_manager")
    
    def get_tools(self) -> List:
        self.logger.info("Loading SQL tools")
        langgraph_tools = LangGraphSQLTools(self.db, self.llm)
        custom_tools = CustomSQLTools(self.db)
        tools = langgraph_tools.get_basic_tools() + custom_tools.get_custom_tools()
        self.logger.info("SQL tools loaded", count=len(tools))
        return tools

# ================================
# SIMPLE TEXT2SQL GRAPH - Agent Based
# ================================
@register_graph("text2sql")
class Text2SQLGraph(BaseGraph):
    """Simple & Clean Text2SQL Graph - LangGraph Native Agents"""
    
    def __init__(self, llm, db=None):
        super().__init__(llm=llm, state_class=Text2SQLState)
        self.logger = log.get(module="text2sql", component="graph")
        
        if db is None:
            db = SQLDatabase.from_uri(config.database.uri)
        
        self.db = db
        self.tool_manager = SQLToolManager(db, llm.get_chat())
        self.tools = self.tool_manager.get_tools() + [analyze_user_query_for_tables]
        
        self._create_agents()
        
        self.logger.info("Text2SQL Graph initialized", 
                        tools_count=len(self.tools),
                        agents=list(self.agents.keys()))
    
    def _create_agents(self):
        """Create all agents with their specific tools"""
        self.agents = {
            "initialization": InitializationAgent(),
            "table_listing": TableListingAgent(self.tools),
            "schema_analysis": SchemaAnalysisAgent(self.tools),
            "schema_retrieval": SchemaRetrievalAgent(self.tools),
            "query_generation": QueryGenerationAgent(self.llm.get_chat(), self.tools),
            "final_answer": FinalAnswerAgent(),
        }
    
    def build_graph(self):
        """Build Simple Agent-Based Graph - Direct Linear Flow"""
        self.logger.info("Building agent-based Text2SQL graph")
        
        memory = MemorySaver()
        graph = StateGraph(Text2SQLState)
        
        # Add all agents as nodes
        for agent_name, agent_instance in self.agents.items():
            graph.add_node(agent_name, agent_instance)
        
        # ================================
        # SIMPLE LINEAR AGENT FLOW
        # ================================
        
        graph.add_edge(START, "initialization")
        graph.add_edge("initialization", "table_listing")
        graph.add_edge("table_listing", "schema_analysis") 
        graph.add_edge("schema_analysis", "schema_retrieval")
        graph.add_edge("schema_retrieval", "query_generation")
        graph.add_edge("query_generation", "final_answer")
        graph.add_edge("final_answer", END)
        
        compiled_graph = graph.compile(
            checkpointer=memory,
            name="agent_text2sql_graph"
        )
        
        self.logger.info("Agent-based Text2SQL graph compiled successfully")
        return compiled_graph