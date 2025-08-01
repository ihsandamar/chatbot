# src/graphs/test_graph.py - Enhanced Test Graph with Table Selection
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Literal, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import BaseModel, Field
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
# QUERY PARAMETER MODELS
# ================================
class FilterCondition(BaseModel):
    """Filter condition for SQL query"""
    field: str = Field(description="Field name to filter on")
    operator: str = Field(description="Filter operator (=, >, <, >=, <=, LIKE, IN, etc.)")
    value: str = Field(description="Filter value")

class OrderBy(BaseModel):
    """Order by clause for SQL query"""
    field: str = Field(description="Field name to order by")
    direction: str = Field(default="ASC", description="Order direction (ASC or DESC)")

class QueryParameters(BaseModel):
    """SQL SELECT query parameters extracted from user prompt"""
    select_fields: List[str] = Field(description="Fields to select (required)")
    group_by_fields: List[str] = Field(default=[], description="Fields to group by")
    order_by: List[OrderBy] = Field(default=[], description="Order by clauses")
    filters: List[FilterCondition] = Field(default=[], description="Filter conditions")
    limit: Optional[int] = Field(default=None, description="Limit number of results")

# ================================
# TABLE MODEL
# ================================
class Table(BaseModel):
    """Table in SQL database with detailed schema information."""
    name: str = Field(description="Name of table in SQL database.")
    columns: List[str] = Field(default=[], description="Column definitions")
    relations: List[str] = Field(default=[], description="Table relations") 
    schema: Optional[str] = Field(default="", description="CREATE TABLE statement")

# ================================
# STATE DEFINITION
# ================================
class BaseState(TypedDict):
    pass

class TestState(BaseState):
    """TestState for the enhanced test graph"""
    user_query: NotRequired[str]  # User's question
    all_tables: NotRequired[List[str]]  # All database tables
    relevant_tables: NotRequired[List[Table]]  # Tables relevant to the user query with schema
    table_schemas: NotRequired[str]  # Raw schema information
    query_parameters: NotRequired[QueryParameters]  # Extracted SQL query parameters

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
        self.logger = log.get(module="test_graph", agent=name)
    
    @abstractmethod
    def invoke(self, state: TestState, config=None) -> Dict[str, Any]:
        """LangGraph Runnable invoke method"""
        pass
    
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a tool by name with error handling"""
        tool = self.tools_dict.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in agent '{self.name}'")
        
        try:
            result = tool.invoke(args)
            self.logger.info("Tool executed successfully", tool=tool_name)
            return result
        except Exception as e:
            self.logger.error("Tool execution failed", tool=tool_name, error=str(e))
            raise e

# ================================
# AGENT IMPLEMENTATIONS
# ================================
class InitializationAgent(BaseAgent):
    """Simple initialization agent"""
    
    def __init__(self):
        super().__init__("InitializationAgent", tools=[])
    
    def invoke(self, state: TestState, config=None) -> Dict[str, Any]:
        self.logger.info("Initializing TestGraph state")
        
        # Set a sample user query if not present
        user_query = state.get("user_query", "Show me all products and their categories")
        
        return {
            "user_query": user_query,
            "all_tables": [],
            "relevant_tables": [],  # List of Table objects
            "table_schemas": "",
            "query_parameters": QueryParameters(select_fields=[])
        }

class TableListingAgent(BaseAgent):
    """Database table listing agent"""
    
    def __init__(self, tools: List):
        # Filter only the sql_db_list_tables tool
        list_tools = [t for t in tools if t.name == "sql_db_list_tables"]
        super().__init__("TableListingAgent", tools=list_tools)
    
    def invoke(self, state: TestState, config=None) -> Dict[str, Any]:
        self.logger.info("Listing database tables")
        
        try:
            # Execute list tables tool
            result = self.execute_tool("sql_db_list_tables", {})
            
            # Parse table list
            all_tables = [t.strip().strip("'\"") for t in str(result).replace('[', '').replace(']', '').replace("'", "").split(',')]
            all_tables = [t for t in all_tables if t]
            
            self.logger.info("Tables listed successfully", count=len(all_tables))
            
            return {"all_tables": all_tables}
            
        except Exception as e:
            self.logger.error("Failed to list tables", error=str(e))
            return {"all_tables": []}

class RelevantTablesAgent(BaseAgent):
    """Agent to find relevant tables using LLM with tool calling (like in the document)"""
    
    def __init__(self, llm):
        super().__init__("RelevantTablesAgent", tools=[])
        self.llm = llm
        self._setup_table_selection_chain()
    
    def _setup_table_selection_chain(self):
        """Setup the table selection chain based on the document approach"""
        # Create the prompt template
        self.system_prompt = """Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools([Table])
        self.output_parser = PydanticToolsParser(tools=[Table])
        
        # Create the chain
        self.table_chain = self.prompt | self.llm_with_tools | self.output_parser
    
    def invoke(self, state: TestState, config=None) -> Dict[str, Any]:
        self.logger.info("Finding relevant tables using LLM")
        
        all_tables = state.get("all_tables", [])
        user_query = state.get("user_query", "")
        
        if not all_tables:
            self.logger.warning("No tables found in state")
            return {"relevant_tables": []}
        
        if not user_query:
            self.logger.warning("No user query found in state")
            return {"relevant_tables": []}
        
        try:
            # Prepare table names string
            table_names = "\n".join(all_tables)
            
            # Invoke the chain
            result = self.table_chain.invoke({
                "table_names": table_names,
                "input": user_query
            })
            
            # Convert to Table objects with empty schema initially
            relevant_tables = [
                Table(name=table.name, columns=[], relations=[], schema="") 
                for table in result if isinstance(table, Table)
            ]
            
            self.logger.info("Relevant tables found using LLM", 
                           count=len(relevant_tables), 
                           tables=[t.name for t in relevant_tables])
            
            return {"relevant_tables": relevant_tables}
            
        except Exception as e:
            self.logger.error("Failed to find relevant tables", error=str(e))
            # Fallback: return first 3 tables as Table objects
            fallback_table_names = all_tables[:3] if len(all_tables) > 3 else all_tables
            fallback_tables = [
                Table(name=name, columns=[], relations=[], schema="") 
                for name in fallback_table_names
            ]
            self.logger.info("Using fallback table selection", tables=[t.name for t in fallback_tables])
            return {"relevant_tables": fallback_tables}
class SchemaAgent(BaseAgent):
    """Agent to get schema for relevant tables and populate Table objects"""
    
    def __init__(self, tools: List):
        # Filter only the sql_db_schema tool
        schema_tools = [t for t in tools if t.name == "sql_db_schema"]
        super().__init__("SchemaAgent", tools=schema_tools)
    
    def _parse_schema_to_table_object(self, table_obj: Table, raw_schema: str) -> Table:
        """Parse raw schema and populate Table object"""
        try:
            # Store full schema
            table_obj.schema = raw_schema
            
            # Simple parsing for columns (basic implementation)
            lines = raw_schema.split('\n')
            columns = []
            relations = []
            
            for line in lines:
                line = line.strip()
                # Look for column definitions (starts with [ and contains column info)
                if line.startswith('[') and '] ' in line and not line.startswith('CONSTRAINT'):
                    columns.append(line)
                # Look for foreign key constraints
                elif 'FOREIGN KEY' in line.upper() or 'REFERENCES' in line.upper():
                    relations.append(line)
            
            table_obj.columns = columns
            table_obj.relations = relations
            
            self.logger.info("Schema parsed for table", 
                           table=table_obj.name, 
                           columns_count=len(columns),
                           relations_count=len(relations))
            
            return table_obj
            
        except Exception as e:
            self.logger.error("Failed to parse schema", table=table_obj.name, error=str(e))
            # Keep original schema at least
            table_obj.schema = raw_schema
            return table_obj
    
    def invoke(self, state: TestState, config=None) -> Dict[str, Any]:
        self.logger.info("Getting schema for relevant tables")
        
        relevant_tables = state.get("relevant_tables", [])
        
        if not relevant_tables:
            self.logger.warning("No relevant tables found, skipping schema retrieval")
            return {"relevant_tables": []}
        
        try:
            # Get table names for schema query
            table_names = ", ".join([table.name for table in relevant_tables])
            
            # Execute schema tool
            raw_schema = self.execute_tool("sql_db_schema", {"table_names": table_names})
            
            # Parse schema and update Table objects
            updated_tables = []
            
            # Split schema by table (simple approach - each CREATE TABLE section)
            schema_text = str(raw_schema)
            
            for table_obj in relevant_tables:
                # Find this table's schema in the raw result
                table_schema_start = schema_text.find(f"CREATE TABLE {table_obj.name}")
                if table_schema_start == -1:
                    table_schema_start = schema_text.find(f"CREATE TABLE [{table_obj.name}]")
                
                if table_schema_start != -1:
                    # Find end of this table's schema
                    next_create = schema_text.find("CREATE TABLE", table_schema_start + 1)
                    if next_create != -1:
                        table_schema = schema_text[table_schema_start:next_create].strip()
                    else:
                        table_schema = schema_text[table_schema_start:].strip()
                    
                    # Parse and update the table object
                    updated_table = self._parse_schema_to_table_object(table_obj, table_schema)
                    updated_tables.append(updated_table)
                else:
                    # Table schema not found, keep original
                    self.logger.warning("Schema not found for table", table=table_obj.name)
                    updated_tables.append(table_obj)
            
            self.logger.info("Schema populated for tables", 
                           count=len(updated_tables))
            
            return {
                "relevant_tables": updated_tables,
                "table_schemas": str(raw_schema)
            }
            
        except Exception as e:
            self.logger.error("Failed to get schema", error=str(e))
            return {"relevant_tables": relevant_tables, "table_schemas": ""}

class QueryParameterAgent(BaseAgent):
    """Agent to extract SQL query parameters from user prompt"""
    
    def __init__(self, llm):
        super().__init__("QueryParameterAgent", tools=[])
        self.llm = llm
        self._setup_parameter_extraction_chain()
    
    def _setup_parameter_extraction_chain(self):
        """Setup the parameter extraction chain using LLM with tool calling"""
        
        self.system_prompt = """You are a SQL query parameter extraction expert. 
Analyze the user's natural language query and extract the following SQL SELECT parameters:

1. **select_fields** (REQUIRED): Which columns/fields should be selected
2. **group_by_fields** (OPTIONAL): Which fields to group by  
3. **order_by** (OPTIONAL): Which fields to order by and direction (ASC/DESC)
4. **filters** (OPTIONAL): Filter conditions with field, operator, and value
5. **limit** (OPTIONAL): Maximum number of results

Available table schemas:
{table_schemas}

Common SQL operators for filters: =, >, <, >=, <=, LIKE, IN, BETWEEN

Extract parameters based ONLY on what the user explicitly requests. Don't assume anything not mentioned.

Use ONLY the available column names from the table schemas provided above.
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "User Query: {user_query}\n\nExtract the SQL parameters from this query."),
        ])
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools([QueryParameters])
        self.output_parser = PydanticToolsParser(tools=[QueryParameters])
        
        # Create the chain
        self.parameter_chain = self.prompt | self.llm_with_tools | self.output_parser
    
    def invoke(self, state: TestState, config=None) -> Dict[str, Any]:
        self.logger.info("Extracting query parameters from user prompt")
        
        user_query = state.get("user_query", "")
        relevant_tables = state.get("relevant_tables", [])
        
        if not user_query:
            self.logger.error("No user query found in state")
            return {"query_parameters": QueryParameters(select_fields=[])}
        
        # Prepare table schemas for context
        table_schemas_text = ""
        if relevant_tables:
            for table in relevant_tables:
                table_schemas_text += f"\nTable: {table.name}\n"
                if table.columns:
                    table_schemas_text += "Columns:\n"
                    for col in table.columns[:10]:  # Limit to first 10 columns for brevity
                        table_schemas_text += f"  {col}\n"
                if len(table.columns) > 10:
                    table_schemas_text += f"  ... and {len(table.columns) - 10} more columns\n"
        else:
            table_schemas_text = "No table schemas available"
        
        try:
            # Invoke the parameter extraction chain
            result = self.parameter_chain.invoke({
                "user_query": user_query,
                "table_schemas": table_schemas_text
            })
            
            if result and len(result) > 0:
                query_params = result[0] if isinstance(result, list) else result
                
                # Validate that select_fields is not empty
                if not query_params.select_fields:
                    self.logger.error("No select fields extracted from user query")
                    return {
                        "query_parameters": QueryParameters(
                            select_fields=["ERROR: Could not determine which fields to select"]
                        )
                    }
                
                self.logger.info("Query parameters extracted successfully", 
                               select_fields=query_params.select_fields,
                               filters_count=len(query_params.filters),
                               group_by_count=len(query_params.group_by_fields),
                               order_by_count=len(query_params.order_by))
                
                return {"query_parameters": query_params}
            else:
                self.logger.error("No parameters extracted from user query")
                return {"query_parameters": QueryParameters(select_fields=[])}
                
        except Exception as e:
            self.logger.error("Failed to extract query parameters", error=str(e))
            return {
                "query_parameters": QueryParameters(
                    select_fields=[f"ERROR: Failed to parse query - {str(e)}"]
                )
            }

# ================================
# TOOL MANAGER
# ================================
class SQLToolManager:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.logger = log.get(module="test_graph", component="tool_manager")
    
    def get_tools(self) -> List:
        self.logger.info("Loading SQL tools")
        langgraph_tools = LangGraphSQLTools(self.db, self.llm)
        custom_tools = CustomSQLTools(self.db)
        tools = langgraph_tools.get_basic_tools() + custom_tools.get_custom_tools()
        self.logger.info("SQL tools loaded", count=len(tools))
        return tools

# ================================
# TEST GRAPH
# ================================
@register_graph("test")
class TestGraph(BaseGraph):
    """Enhanced TestGraph with intelligent table selection"""
    
    def __init__(self, llm, db=None):
        super().__init__(llm=llm, state_class=TestState)
        self.logger = log.get(module="test_graph", component="graph")
        
        if db is None:
            db = SQLDatabase.from_uri(config.database.uri)
        
        self.db = db
        self.tool_manager = SQLToolManager(db, llm.get_chat())
        self.tools = self.tool_manager.get_tools()
        
        self._create_agents()
        
        self.logger.info("Enhanced TestGraph initialized", 
                        tools_count=len(self.tools),
                        agents=list(self.agents.keys()))
    
    def _create_agents(self):
        """Create all agents with their specific tools"""
        self.agents = {
            "initialization": InitializationAgent(),
            "table_listing": TableListingAgent(self.tools),
            "relevant_tables": RelevantTablesAgent(self.llm.get_chat()),
            "schema": SchemaAgent(self.tools),
            "query_parameters": QueryParameterAgent(self.llm.get_chat()),
        }
    
    def build_graph(self):
        """Build Enhanced Test Graph - Five Step Flow"""
        self.logger.info("Building enhanced test graph with query parameter extraction")
        
        memory = MemorySaver()
        graph = StateGraph(TestState)
        
        # Add agents as nodes
        for agent_name, agent_instance in self.agents.items():
            graph.add_node(agent_name, agent_instance)
        
        # ================================
        # FIVE-STEP FLOW
        # ================================
        
        graph.add_edge(START, "initialization")
        graph.add_edge("initialization", "table_listing")
        graph.add_edge("table_listing", "relevant_tables")
        graph.add_edge("relevant_tables", "schema")
        graph.add_edge("schema", "query_parameters")
        graph.add_edge("query_parameters", END)
        
        compiled_graph = graph.compile(
            checkpointer=memory,
            name="enhanced_test_graph"
        )
        
        self.logger.info("Enhanced test graph compiled successfully with query parameter extraction")
        return compiled_graph