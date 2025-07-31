# src/tools/langgraph_sql_tools.py
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from typing import List

class LangGraphSQLTools:
    """LangGraph tabanlı SQL araçları"""
    
    def __init__(self, db: SQLDatabase, llm):
        self.db = db
        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
    def get_basic_tools(self) -> List[Tool]:
        """Temel SQL araçlarını döndür"""
        tools = self.toolkit.get_tools()
        return [
            self._get_tool_by_name(tools, "sql_db_list_tables"),
            self._get_tool_by_name(tools, "sql_db_schema"),
            self._get_tool_by_name(tools, "sql_db_query")
        ]
    
    def _get_tool_by_name(self, tools: List[Tool], name: str) -> Tool:
        return next(tool for tool in tools if tool.name == name)