# src/nodes/sql_nodes.py
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.models.models import State
from src.prompts.text2sql_prompts import QUERY_GENERATION_SYSTEM, QUERY_CHECK_SYSTEM

class SQLNodes:
    """SQL işlem nodeları"""
    
    def __init__(self, llm, db, tools):
        self.llm = llm
        self.db = db
        self.tools = tools
        # Tools'ları name ile erişilebilir hale getir
        self.tools_dict = {tool.name: tool for tool in tools}
        self._setup_models()
    
    def _setup_models(self):
        """Model ve prompt'ları hazırla"""
        from src.tools.custom_sql_tools import SubmitFinalAnswer
        
        query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_GENERATION_SYSTEM), 
            ("placeholder", "{messages}")
        ])
        
        query_check_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_CHECK_SYSTEM), 
            ("placeholder", "{messages}")
        ])
        
        self.query_gen = query_gen_prompt | self.llm.bind_tools([SubmitFinalAnswer])
        self.query_check = query_check_prompt | self.llm.bind_tools(
            [tool for tool in self.tools if tool.name == "db_query_tool"], 
            tool_choice="required"
        )
    
    def first_tool_call_node(self, state: State) -> dict:
        """İlk araç çağrısı - tabloları listele"""
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }]
                )
            ]
        }
    
    def schema_node(self, state: State) -> dict:
        """Şema bilgilerini al - TOOL CALL ID HATASI DÜZELTİLDİ"""
        messages = state.get("messages", [])
        
        # Bekleyen tool call'ları kontrol et ve çalıştır
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            new_messages = []
            for tool_call in messages[-1].tool_calls:
                tool_name = tool_call["name"]
                tool_id = tool_call["id"]
                
                if tool_name == "sql_db_list_tables":
                    # List tables tool'unu çalıştır
                    list_tables_tool = self.tools_dict.get("sql_db_list_tables")
                    if list_tables_tool:
                        try:
                            result = list_tables_tool.invoke(tool_call["args"])
                            new_messages.append(
                                ToolMessage(content=result, tool_call_id=tool_id)
                            )
                            print(f"[DEBUG] Tables listed: {result}")
                        except Exception as e:
                            new_messages.append(
                                ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id)
                            )
            
            # Yeni tool message'ları ekle
            messages.extend(new_messages)
        
        # Table listesini bul ve schema iste
        table_list_content = ""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and "sql_db_list_tables" in str(msg.tool_call_id):
                table_list_content = msg.content
                break
        
        if table_list_content:
            # Basit şekilde ilk birkaç tabloyu al
            available_tables = [t.strip() for t in table_list_content.replace('[', '').replace(']', '').split(',')]
            tables_to_query = available_tables[:3] if len(available_tables) > 3 else available_tables
            
            print(f"[DEBUG] Selected tables for schema: {tables_to_query}")
            
            if tables_to_query:
                # Schema için yeni tool call ID oluştur
                schema_call_id = f"schema_call_{len(messages)}"
                
                # Schema request AI message oluştur
                schema_request = AIMessage(
                    content="Getting schema information for the tables.",
                    tool_calls=[{
                        "name": "sql_db_schema",
                        "args": {"table_names": ", ".join(tables_to_query)},
                        "id": schema_call_id,
                    }]
                )
                messages.append(schema_request)
                
                # Hemen schema tool'unu çalıştır
                schema_tool = self.tools_dict.get("sql_db_schema")
                if schema_tool:
                    try:
                        schema_result = schema_tool.invoke({"table_names": ", ".join(tables_to_query)})
                        messages.append(
                            ToolMessage(
                                content=schema_result, 
                                tool_call_id=schema_call_id
                            )
                        )
                        print(f"[DEBUG] Schema retrieved successfully")
                    except Exception as e:
                        print(f"[ERROR] Schema tool failed: {e}")
                        messages.append(
                            ToolMessage(
                                content=f"Error getting schema: {str(e)}", 
                                tool_call_id=schema_call_id
                            )
                        )
        
        return {"messages": messages}
    
    def query_generation_node(self, state: State) -> dict:
        """SQL sorgusu üret"""
        messages = state.get("messages", [])
        message = self.query_gen.invoke({"messages": messages})
        
        # Yanlış araç çağrısı kontrolü
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: Wrong tool called: {tc['name']}. Please use SubmitFinalAnswer only.",
                            tool_call_id=tc["id"],
                        )
                    )
        
        return {"messages": messages + [message] + tool_messages}
    
    def query_check_node(self, state: State) -> dict:
        """Sorguyu kontrol et"""
        last_message = state["messages"][-1]
        result = self.query_check.invoke({"messages": [last_message]})
        return {"messages": state["messages"] + [result]}
    
    def query_execution_node(self, state: State) -> dict:
        """Sorguyu çalıştır"""
        messages = state.get("messages", [])
        
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            for tool_call in messages[-1].tool_calls:
                if tool_call["name"] == "db_query_tool":
                    query = tool_call["args"].get("query", "")
                    if query:
                        print(f"[DEBUG] Executing query: {query}")
                        result = self.db.run_no_throw(query)
                        if not result:
                            result = "Error: Query failed. Please rewrite your query and try again."
                        
                        messages.append(
                            ToolMessage(content=result, tool_call_id=tool_call["id"])
                        )
        
        return {"messages": messages}