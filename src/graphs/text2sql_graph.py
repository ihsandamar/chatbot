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
from src.models.models import LLM, State
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


class SQLExecutionRouter:
    """Router to decide whether to fix SQL or end the flow"""
    
    def __init__(self):
        self.logger = log.get(module="test_graph", component="sql_router")
    
    def route(self, state: State) -> str:
        """Route based on SQL execution result"""
        is_error = state.get("is_error", False)
        
        if is_error:
            self.logger.info("SQL execution failed, routing to fix_sql")
            return "fix_sql"
        else:
            self.logger.info("SQL execution successful, routing to end")
            return "end"

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
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
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
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
        self.logger.info("Initializing TestGraph state")
        
        # Extract user query from messages or use existing user_query
        user_query = state.get("user_query", "")
        
        # If no user_query, extract from latest human message
        if not user_query:
            messages = state.get("messages", [])
            for message in reversed(messages):
                if hasattr(message, 'type') and message.type == "human":
                    # Extract text content from message
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            # Handle content array format
                            for content_part in message.content:
                                if isinstance(content_part, dict) and content_part.get('type') == 'text':
                                    user_query = content_part.get('text', '')
                                    break
                        elif isinstance(message.content, str):
                            user_query = message.content
                        break
        
        # Fallback default query
        if not user_query:
            user_query = "Show me all products and their categories"
        
        self.logger.info("Extracted user query", query=user_query)
        
        return {
            "messages": state.get("messages", []),  # Preserve existing messages
            "user_query": user_query,
            "all_tables": [],
            "relevant_tables": [],  # List of Table objects
            "table_schemas": "",
            "query_parameters": QueryParameters(select_fields=[]),
            "generated_sql": "",
            "validated_sql": "",
            "is_valid": False,
            "sql_result": "",
            "is_error": False,
            "error_message": "",
            "fixed_sql": ""
        }

class TableListingAgent(BaseAgent):
    """Database table listing agent"""
    
    def __init__(self, tools: List):
        # Filter only the sql_db_list_tables tool
        list_tools = [t for t in tools if t.name == "sql_db_list_tables"]
        super().__init__("TableListingAgent", tools=list_tools)
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
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
        # Create the prompt template - Much more inclusive
        self.system_prompt = """You are a database expert helping to identify ALL relevant tables for a query.

Available tables:
{table_names}

CRITICAL RULES:
1. **INCLUDE ALL RELATED TABLES** - Don't just pick the obvious ones
2. **THINK ABOUT RELATIONSHIPS** - If query mentions categories and products, include ALL related tables:
   - Main tables (Products, Categories)
   - Junction/bridge tables (ProductCategories)  
   - Lookup tables that might be needed
3. **BE EXTREMELY INCLUSIVE** - It's better to include too many than miss critical ones
4. **CONSIDER JOINS** - Think about what tables you'd need to JOIN to answer the query
5. **INCLUDE SUPPORTING TABLES** - Any table that might contain referenced data

For the user's query, return ALL tables that could possibly be relevant, including:
- Primary tables mentioned in the query
- Bridge/junction tables for many-to-many relationships
- Lookup tables with names/descriptions
- Any table that might be involved in JOINs

Example: If query asks for "product names by category", include:
- Products (main table)
- Categories (lookup table) 
- ProductCategories (bridge table)
- Any other product-related tables

Be VERY generous in your selection - missing a table is worse than including an extra one."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools([Table])
        self.output_parser = PydanticToolsParser(tools=[Table])
        
        # Create the chain
        self.table_chain = self.prompt | self.llm_with_tools | self.output_parser
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
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
            
            # Extract table names from result and filter against actual tables
            suggested_tables = [table.name for table in result if isinstance(table, Table)]
            
            # CRITICAL FIX: Only include tables that actually exist in all_tables
            valid_tables = []
            for suggested_table in suggested_tables:
                if suggested_table in all_tables:
                    valid_tables.append(suggested_table)
                else:
                    self.logger.warning("LLM suggested non-existent table", 
                                      suggested=suggested_table, 
                                      available=len(all_tables))
            
            # Convert to Table objects with empty schema initially
            relevant_tables = [
                Table(name=table_name, columns=[], relations=[], schema="") 
                for table_name in valid_tables
            ]
            
            self.logger.info("Relevant tables found and filtered", 
                           suggested_count=len(suggested_tables),
                           valid_count=len(relevant_tables), 
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
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
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
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
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

class SQLGeneratorAgent(BaseAgent):
    """AI-driven agent to generate SQL query from extracted parameters and table information"""
    
    def __init__(self, llm):
        super().__init__("SQLGeneratorAgent", tools=[])
        self.llm = llm
        self._setup_sql_generation_chain()
    
    def _setup_sql_generation_chain(self):
        """Setup the SQL generation chain using LLM"""
        
        self.system_prompt = """You are a MSSQL expert. Generate a syntactically correct MSSQL SELECT query based on the provided parameters.

Table Schemas:
{table_schemas}

Query Parameters:
- Select Fields: {select_fields}
- Group By Fields: {group_by_fields}  
- Order By: {order_by}
- Filters: {filters}
- Limit: {limit}

MSSQL Rules:
- Use TOP {limit} instead of LIMIT for row limiting
- Use proper MSSQL syntax and functions
- Use appropriate JOIN syntax when multiple tables are involved
- Use square brackets [TableName] or [ColumnName] when needed
- Handle data types correctly (NVARCHAR, UNIQUEIDENTIFIER, etc.)

Generate ONLY a valid MSSQL SELECT statement. No explanations or additional text.
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Generate the SQL query based on the parameters above."),
        ])
        
        # Create the chain
        self.sql_chain = self.prompt | self.llm
    
    def _format_parameters_for_prompt(self, query_parameters: QueryParameters) -> Dict[str, str]:
        """Format query parameters for the prompt"""
        
        # Format order by
        order_by_str = ""
        if query_parameters.order_by:
            order_items = [f"{ob.field} {ob.direction}" for ob in query_parameters.order_by]
            order_by_str = ", ".join(order_items)
        
        # Format filters
        filters_str = ""
        if query_parameters.filters:
            filter_items = [f"{f.field} {f.operator} {f.value}" for f in query_parameters.filters]
            filters_str = " AND ".join(filter_items)
        
        return {
            "select_fields": ", ".join(query_parameters.select_fields) if query_parameters.select_fields else "*",
            "group_by_fields": ", ".join(query_parameters.group_by_fields) if query_parameters.group_by_fields else "None",
            "order_by": order_by_str if order_by_str else "None",
            "filters": filters_str if filters_str else "None",
            "limit": str(query_parameters.limit) if query_parameters.limit else "None"
        }
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
        self.logger.info("Generating SQL query using AI")
        
        query_parameters = state.get("query_parameters")
        relevant_tables = state.get("relevant_tables", [])
        
        if not query_parameters:
            self.logger.error("No query parameters found in state")
            return {"generated_sql": "-- ERROR: No query parameters found"}
        
        if not relevant_tables:
            self.logger.error("No relevant tables found in state")
            return {"generated_sql": "-- ERROR: No relevant tables found"}
        
        try:
            # Prepare table schemas for prompt
            table_schemas_text = ""
            for table in relevant_tables:
                table_schemas_text += f"\nTable: {table.name}\n"
                if table.schema:
                    # Show first few lines of schema
                    schema_lines = table.schema.split('\n')[:10]
                    table_schemas_text += "\n".join(schema_lines)
                    if len(table.schema.split('\n')) > 10:
                        table_schemas_text += "\n... (schema truncated)"
                elif table.columns:
                    table_schemas_text += "Columns:\n"
                    for col in table.columns[:15]:  # Limit to first 15 columns
                        table_schemas_text += f"  {col}\n"
                    if len(table.columns) > 15:
                        table_schemas_text += f"  ... and {len(table.columns) - 15} more columns\n"
                table_schemas_text += "\n"
            
            # Format parameters
            formatted_params = self._format_parameters_for_prompt(query_parameters)
            
            # Invoke the AI chain
            response = self.sql_chain.invoke({
                "table_schemas": table_schemas_text,
                **formatted_params
            })
            
            # Extract SQL from response
            generated_sql = response.content if hasattr(response, 'content') else str(response)
            generated_sql = generated_sql.strip()
            
            # Clean up any markdown formatting
            if generated_sql.startswith("```sql"):
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
            elif generated_sql.startswith("```"):
                generated_sql = generated_sql.replace("```", "").strip()
            
            self.logger.info("SQL query generated successfully using AI", 
                           sql_length=len(generated_sql),
                           tables_count=len(relevant_tables))
            
            return {"generated_sql": generated_sql}
            
        except Exception as e:
            self.logger.error("Failed to generate SQL query using AI", error=str(e))
            return {"generated_sql": f"-- ERROR: Failed to generate SQL using AI - {str(e)}"}

class SQLValidatorAgent(BaseAgent):
    """Agent to validate that generated SQL is a safe SELECT statement"""
    
    def __init__(self):
        super().__init__("SQLValidatorAgent", tools=[])
    
    def _is_safe_select_query(self, sql: str) -> tuple[bool, str]:
        """Check if SQL is a safe SELECT query"""
        
        if not sql or sql.startswith("-- ERROR"):
            return False, "SQL is empty or contains errors"
        
        # Clean and normalize SQL
        cleaned_sql = sql.strip().upper()
        
        # Remove comments
        lines = []
        for line in cleaned_sql.split('\n'):
            if not line.strip().startswith('--'):
                lines.append(line)
        cleaned_sql = ' '.join(lines)
        
        # Check if it starts with SELECT
        if not cleaned_sql.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Forbidden keywords (DML/DDL operations)
        forbidden_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'DECLARE', 'SET', 'GRANT', 
            'REVOKE', 'DENY', 'BACKUP', 'RESTORE', 'MERGE', 'BULK'
        ]
        
        for keyword in forbidden_keywords:
            if f' {keyword} ' in f' {cleaned_sql} ':
                return False, f"Forbidden keyword detected: {keyword}"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'EXEC(',
            'EXECUTE(',
            'SP_',
            'XP_',
            'OPENROWSET',
            'OPENDATASOURCE'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in cleaned_sql:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, "Valid SELECT query"
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
        self.logger.info("Validating generated SQL query")
        
        generated_sql = state.get("generated_sql", "")
        
        if not generated_sql:
            self.logger.error("No generated SQL found in state")
            return {
                "validated_sql": "-- ERROR: No SQL to validate",
                "is_valid": False
            }
        
        try:
            is_valid, message = self._is_safe_select_query(generated_sql)
            
            if is_valid:
                self.logger.info("SQL validation passed", message=message)
                return {
                    "validated_sql": generated_sql,
                    "is_valid": True
                }
            else:
                self.logger.error("SQL validation failed", reason=message)
                return {
                    "validated_sql": f"-- VALIDATION ERROR: {message}\n-- Original SQL: {generated_sql}",
                    "is_valid": False
                }
                
        except Exception as e:
            self.logger.error("SQL validation error", error=str(e))
            return {
                "validated_sql": f"-- VALIDATION ERROR: {str(e)}\n-- Original SQL: {generated_sql}",
                "is_valid": False
            }

class SQLExecutorAgent(BaseAgent):
    """Agent to execute validated SQL queries using database tools"""
    
    def __init__(self, tools: List):
        # Filter only the db_query_tool for SQL execution
        execution_tools = [t for t in tools if t.name == "db_query_tool"]
        super().__init__("SQLExecutorAgent", tools=execution_tools)
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
        self.logger.info("Executing validated SQL query")
        
        validated_sql = state.get("validated_sql", "")
        is_valid = state.get("is_valid", False)
        
        if not validated_sql:
            self.logger.error("No validated SQL found in state")
            return {
                "sql_result": "ERROR: No SQL to execute",
                "is_error": True,
                "error_message": "No validated SQL found in state"
            }
        
        if not is_valid:
            self.logger.error("Cannot execute invalid SQL")
            return {
                "sql_result": "ERROR: Cannot execute invalid SQL",
                "is_error": True,
                "error_message": "Cannot execute invalid SQL"
            }
        
        # Check if SQL starts with validation error
        if validated_sql.startswith("-- VALIDATION ERROR"):
            self.logger.error("Cannot execute SQL with validation errors")
            return {
                "sql_result": "ERROR: Cannot execute SQL with validation errors",
                "is_error": True,
                "error_message": "Cannot execute SQL with validation errors"
            }
        
        try:
            # Execute the SQL using db_query_tool
            result = self.execute_tool("db_query_tool", {"query": validated_sql})
            
            self.logger.info("SQL executed successfully", 
                           result_length=len(str(result)))
            
            # Format the result for better display
            sql_result = str(result)
            formatted_message = f"SQL query executed successfully!\n\nQuery: {validated_sql}\n\nResult:\n{sql_result}"
            
            return {
                "messages": [AIMessage(content=formatted_message)],
                "sql_result": sql_result,
                "is_error": False,
                "error_message": ""
            }
            
        except Exception as e:
            self.logger.error("Failed to execute SQL", error=str(e))
            return {
                "sql_result": f"EXECUTION ERROR: {str(e)}",
                "is_error": True,
                "error_message": str(e)
            }

class FixSQLAgent(BaseAgent):
    """AI-powered agent to fix SQL execution errors"""
    
    def __init__(self, llm):
        super().__init__("FixSQLAgent", tools=[])
        self.llm = llm
        self._setup_fix_sql_chain()
    
    def _setup_fix_sql_chain(self):
        """Setup the SQL fixing chain using LLM"""
        
        self.system_prompt = """You are a MSSQL expert specialized in fixing SQL execution errors.

You will receive:
1. **Original SQL Query**: The SQL that failed to execute
2. **Error Message**: The specific database error 
3. **Table Schemas**: Available table structures with correct column names
4. **User Query**: The original user request

Your task:
- Analyze the error message carefully
- Check table schemas for correct column names and data types
- Fix the SQL query to address the specific error
- Use only columns that actually exist in the table schemas
- Maintain the original query intent and logic
- Use proper MSSQL syntax

Common MSSQL errors to fix:
- Invalid column names (use correct column names from schema)
- Invalid table references  
- JOIN syntax errors (use correct foreign key relationships)
- Data type mismatches
- Syntax errors

Table Schemas:
{table_schemas}

Original SQL:
{original_sql}

Error Message:
{error_message}

User Query Context:
{user_query}

Generate ONLY a corrected MSSQL SELECT statement. No explanations or additional text.
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Fix the SQL query based on the error message and table schemas above."),
        ])
        
        # Create the chain
        self.fix_chain = self.prompt | self.llm
    
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
        self.logger.info("Fixing SQL execution error using AI")
        
        validated_sql = state.get("validated_sql", "")
        error_message = state.get("error_message", "")
        relevant_tables = state.get("relevant_tables", [])
        user_query = state.get("user_query", "")
        is_error = state.get("is_error", False)
        
        if not is_error:
            self.logger.info("No error to fix, returning original SQL")
            return {"fixed_sql": validated_sql}
        
        if not validated_sql or not error_message:
            self.logger.error("Missing SQL or error message for fixing")
            return {"fixed_sql": "-- ERROR: Cannot fix SQL - missing information"}
        
        try:
            # Prepare table schemas for prompt
            table_schemas_text = ""
            if relevant_tables:
                for table in relevant_tables:
                    table_schemas_text += f"\nTable: {table.name}\n"
                    if table.schema:
                        # Show full schema for fixing
                        table_schemas_text += table.schema + "\n"
                    elif table.columns:
                        table_schemas_text += "Columns:\n"
                        for col in table.columns:
                            table_schemas_text += f"  {col}\n"
                    table_schemas_text += "\n"
            else:
                table_schemas_text = "No table schemas available"
            
            # Invoke the AI fix chain
            response = self.fix_chain.invoke({
                "table_schemas": table_schemas_text,
                "original_sql": validated_sql,
                "error_message": error_message,
                "user_query": user_query
            })
            
            # Extract fixed SQL from response
            fixed_sql = response.content if hasattr(response, 'content') else str(response)
            fixed_sql = fixed_sql.strip()
            
            # Clean up any markdown formatting
            if fixed_sql.startswith("```sql"):
                fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
            elif fixed_sql.startswith("```"):
                fixed_sql = fixed_sql.replace("```", "").strip()
            
            self.logger.info("SQL fixed successfully using AI", 
                           original_length=len(validated_sql),
                           fixed_length=len(fixed_sql))
            
            formatted_message = f"SQL error has been fixed!\n\nOriginal Error: {error_message}\n\nFixed SQL: {fixed_sql}"
            
            return {
                "messages": [AIMessage(content=formatted_message)],
                "fixed_sql": fixed_sql
            }
            
        except Exception as e:
            self.logger.error("Failed to fix SQL using AI", error=str(e))
            return {
                "messages": [AIMessage(content=f"ERROR: Failed to fix SQL - {str(e)}")],
                "fixed_sql": f"-- ERROR: Failed to fix SQL - {str(e)}"
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
# Text2SQL GRAPH
# ================================
@register_graph("text2sql")
class Text2SQLGraph(BaseGraph):
    """Enhanced Text2SQLGraph with intelligent table selection"""

    def __init__(self, llm: LLM, db=None):
        super().__init__(llm=llm, state_class=State)
        self.logger = log.get(module="test_graph", component="graph")
        
        if db is None:
            db = SQLDatabase.from_uri(config.database.uri)
        
        self.db = db
        self.tool_manager = SQLToolManager(db, llm.get_chat())
        self.tools = self.tool_manager.get_tools()
        
        self._create_agents()
        
        # Build and store the compiled graph for agent interface
        self._compiled_graph = None
        
        self.logger.info("Enhanced TestGraph initialized", 
                        tools_count=len(self.tools),
                        agents=list(self.agents.keys()))
    
    # Agent interface for supervisor compatibility
    def invoke(self, state: State, config=None) -> Dict[str, Any]:
        """Agent interface - invoke the full text2sql workflow"""
        try:
            # Build graph if not already built
            if self._compiled_graph is None:
                self._compiled_graph = self.build_graph()
            
            # Execute the full text2sql workflow
            result = self._compiled_graph.invoke(state, config=config)
            
            # Return the final state
            return result
            
        except Exception as e:
            self.logger.error("Text2SQL workflow failed", error=str(e))
            return {
                "messages": [AIMessage(content=f"ERROR: Text2SQL workflow failed - {str(e)}")],
                "is_error": True,
                "error_message": str(e)
            }
    
    def _create_agents(self):
        """Create all agents with their specific tools"""
        self.agents = {
            "initialization": InitializationAgent(),
            "table_listing": TableListingAgent(self.tools),
            "relevant_tables_agent": RelevantTablesAgent(self.llm.get_chat()),
            "schema_agent": SchemaAgent(self.tools),
            "query_parameters_agent": QueryParameterAgent(self.llm.get_chat()),
            "sql_generator": SQLGeneratorAgent(self.llm.get_chat()),
            "sql_validator": SQLValidatorAgent(),
            "sql_executor": SQLExecutorAgent(self.tools),
            "fix_sql_agent": FixSQLAgent(self.llm.get_chat()),
        }
        
        # Create router
        self.sql_router = SQLExecutionRouter()
    
    def build_graph(self):
        """Build Enhanced Text2SQL Graph - Nine Step Flow with Error Handling and SQL Fixing"""
        self.logger.info("Building enhanced text2sql graph with SQL error handling and fixing")
        
        memory = MemorySaver()
        graph = StateGraph(State)
        
        # Add agents as nodes
        for agent_name, agent_instance in self.agents.items():
            graph.add_node(agent_name, agent_instance)
        
        # ================================
        # NINE-STEP FLOW WITH CONDITIONAL ROUTING
        # ================================
        
        graph.add_edge(START, "initialization")
        graph.add_edge("initialization", "table_listing")
        graph.add_edge("table_listing", "relevant_tables_agent")
        graph.add_edge("relevant_tables_agent", "schema_agent")
        graph.add_edge("schema_agent", "query_parameters_agent")
        graph.add_edge("query_parameters_agent", "sql_generator")
        graph.add_edge("sql_generator", "sql_validator")
        graph.add_edge("sql_validator", "sql_executor")
        
        # Conditional routing after SQL execution
        graph.add_conditional_edges(
            "sql_executor",
            self.sql_router.route,
            {
                "fix_sql_agent": "fix_sql_agent",
                "end": END
            }
        )
        
        # Fix SQL agent goes to END
        graph.add_edge("fix_sql_agent", END)
        
        compiled_graph = graph.compile(
            checkpointer=memory,
            name="text2sql_graph"
        )

        self.logger.info("Enhanced text2sql graph compiled successfully with error handling and SQL fixing")
        return compiled_graph