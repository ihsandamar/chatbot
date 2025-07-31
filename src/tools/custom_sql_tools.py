# src/tools/custom_sql_tools.py
from langchain_core.tools import Tool
from functools import partial
from pydantic import BaseModel, Field

class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

def db_query_function(query: str, database) -> str:
    """Execute SQL query on database with error handling."""
    result = database.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result

def create_db_query_tool(database) -> Tool:
    """Create a database query tool with partial function binding."""
    return Tool.from_function(
        func=partial(db_query_function, database=database),
        name="db_query_tool",
        description="Executes a SQL query on the database and returns the result or an error message.",
    )

class CustomSQLTools:
    """Özel SQL araçları"""
    
    def __init__(self, database):
        self.database = database
    
    def get_custom_tools(self) -> list:
        """Özel araçları döndür"""
        return [
            create_db_query_tool(self.database),
            # Diğer özel araçlar buraya eklenebilir
        ]