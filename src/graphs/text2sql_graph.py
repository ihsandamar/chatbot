# src/graphs/text2sql_graph.py - SOLID Refactored Version
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import State
from src.tools.langgraph_sql_tools import LangGraphSQLTools
from src.tools.custom_sql_tools import CustomSQLTools


# SOLID Principle 1: Single Responsibility Principle (SRP)
class MessageExtractor:
    """Mesajlardan bilgi çıkarma sorumluluğu - SRP"""
    
    @staticmethod
    def get_last_user_question(messages: List) -> str:
        """Son kullanıcı sorusunu çıkar"""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list) and len(msg.content) > 0:
                        if isinstance(msg.content[0], dict) and 'text' in msg.content[0]:
                            return msg.content[0]['text']
                    elif isinstance(msg.content, str):
                        return msg.content
        return "Soru bulunamadı"
    
    @staticmethod
    def find_tool_result(messages: List, tool_call_pattern: str) -> tuple[str, str]:
        """Tool sonucunu ve çalıştırılan query'yi bul"""
        query_result = ""
        executed_query = ""
        
        for msg in reversed(messages):
            if hasattr(msg, 'tool_call_id') and hasattr(msg, 'content'):
                if tool_call_pattern in str(msg.tool_call_id):
                    query_result = msg.content
                    
                    # Önceki AI message'da query var
                    msg_index = messages.index(msg)
                    if msg_index > 0:
                        prev_msg = messages[msg_index - 1]
                        if hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for tc in prev_msg.tool_calls:
                                if tc.get("name") == "db_query_tool":
                                    executed_query = tc["args"].get("__arg1", "") or tc["args"].get("query", "")
                                    break
                    break
        
        return query_result, executed_query


class TableSelector:
    """Tablo seçme sorumluluğu - SRP"""
    
    @staticmethod
    def select_relevant_tables(user_question: str, available_tables: List[str]) -> List[str]:
        """Kullanıcı sorusuna göre ilgili tabloları seç"""
        user_question_lower = user_question.lower()
        
        if "product" in user_question_lower or "espresso" in user_question_lower:
            return ["Products", "Categories", "ProductCategories"]
        elif "order" in user_question_lower:
            return ["Orders", "OrderDetails", "Products", "Customers"]
        elif "customer" in user_question_lower:
            return ["Customers", "Orders"]
        else:
            # Default olarak ilk 3 tabloyu al
            return available_tables[:3]


# SOLID Principle 2: Open/Closed Principle (OCP)
class NodeInterface(ABC):
    """Node'lar için ortak interface - OCP"""
    
    @abstractmethod
    def execute(self, state: State) -> Dict[str, Any]:
        pass


class BaseNode(NodeInterface):
    """Node'lar için base class - OCP"""
    
    def __init__(self, name: str):
        self.name = name
    
    def log_debug(self, message: str):
        print(f"[DEBUG] {self.name}: {message}")


class ListTablesNode(BaseNode):
    """Tabloları listeleme node'u - SRP"""
    
    def __init__(self):
        super().__init__("ListTablesNode")
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Listing all database tables")
        messages = state["messages"]
        
        response = AIMessage(
            content="Listing all database tables to understand the schema.",
            tool_calls=[{
                "name": "sql_db_list_tables",
                "args": {},
                "id": "list_tables_call",
            }]
        )
        
        return {"messages": messages + [response]}


class GetSchemaNode(BaseNode):
    """Şema alma node'u - SRP"""
    
    def __init__(self):
        super().__init__("GetSchemaNode")
        self.message_extractor = MessageExtractor()
        self.table_selector = TableSelector()
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Getting table schemas")
        messages = state["messages"]
        
        # Table listesini bul
        table_list = ""
        for msg in reversed(messages):
            if hasattr(msg, 'tool_call_id') and "list_tables" in str(msg.tool_call_id):
                table_list = msg.content
                break
        
        if table_list:
            # Kullanıcı sorusunu al ve ilgili tabloları seç
            user_question = self.message_extractor.get_last_user_question(messages)
            available_tables = [t.strip() for t in table_list.replace('[', '').replace(']', '').split(',')]
            important_tables = self.table_selector.select_relevant_tables(user_question, available_tables)
            
            self.log_debug(f"Selected tables: {important_tables}")
            
            response = AIMessage(
                content=f"Getting schema for relevant tables: {', '.join(important_tables)}",
                tool_calls=[{
                    "name": "sql_db_schema",
                    "args": {"table_names": ", ".join(important_tables)},
                    "id": "schema_call",
                }]
            )
            
            return {"messages": messages + [response]}
        
        # Table list yoksa boş response
        response = AIMessage(content="No tables found to get schema.")
        return {"messages": messages + [response]}


class ExecuteQueryNode(BaseNode):
    """Query çalıştırma node'u - SRP"""
    
    def __init__(self, llm, tools):
        super().__init__("ExecuteQueryNode")
        self.query_model = self._create_query_model(llm, tools)
    
    def _create_query_model(self, llm, tools):
        """Query model oluştur"""
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MSSQL expert. Based on the table schemas provided, generate a syntactically correct MSSQL query and execute it.

Rules:
- Use the exact table and column names from the schemas
- Join tables properly using foreign keys
- Get at most 5 results unless specified otherwise
- Only query relevant columns
- Use meaningful ORDER BY clauses
- Handle many-to-many relationships through junction tables

After generating the SQL query, IMMEDIATELY use db_query_tool to execute it."""),
            ("placeholder", "{messages}")
        ])
        
        return query_prompt | llm.bind_tools([
            tool for tool in tools if tool.name == "db_query_tool"
        ], tool_choice="required")
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Generate and execute query")
        messages = state["messages"]
        
        # Schema var mı kontrol et
        schema_found = any(
            hasattr(msg, 'tool_call_id') and "schema" in str(msg.tool_call_id) 
            for msg in messages
        )
        
        if schema_found:
            response = self.query_model.invoke({"messages": messages})
            self.log_debug(f"Query model response with {len(getattr(response, 'tool_calls', []))} tool calls")
            return {"messages": messages + [response]}
        else:
            response = AIMessage(content="Schema not found, cannot generate query.")
            return {"messages": messages + [response]}


class FinalAnswerNode(BaseNode):
    """Final cevap node'u - SRP"""
    
    def __init__(self):
        super().__init__("FinalAnswerNode")
        self.message_extractor = MessageExtractor()
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Creating final answer")
        messages = state["messages"]
        
        # Kullanıcı sorusunu al
        user_question = self.message_extractor.get_last_user_question(messages)
        
        # Query sonucunu bul
        query_result, executed_query = self.message_extractor.find_tool_result(messages, "call_")
        
        if query_result and not query_result.startswith("Error:") and query_result != "":
            final_text = f"""✅ **Query Başarıyla Çalıştırıldı!**

📋 **Sorunuz:** {user_question}

🔍 **Çalıştırılan SQL:**
```sql
{executed_query}
```

📊 **Sonuçlar:**
{query_result}

💡 **Özet:** Sorgunuz başarıyla çalıştırıldı ve sonuçlar yukarıda görüntülendi."""
        else:
            final_text = f"""❌ **Query Hatası**

📋 **Sorunuz:** {user_question}

Query sonucu bulunamadı veya hata oluştu. Lütfen tekrar deneyin."""
        
        response = AIMessage(content=final_text)
        self.log_debug(f"Final answer created for question: {user_question[:50]}...")
        return {"messages": messages + [response]}


# SOLID Principle 3: Liskov Substitution Principle (LSP)
class RouterInterface(ABC):
    """Router'lar için interface - LSP"""
    
    @abstractmethod
    def route(self, state: State) -> str:
        pass


class SimpleRouter(RouterInterface):
    """Basit routing implementasyonu - LSP"""
    
    def route(self, state: State) -> str:
        messages = state["messages"]
        
        print(f"[DEBUG] Simple routing, messages: {len(messages)}")
        
        # Son 2 mesajı kontrol et
        if len(messages) >= 2:
            last_msg = messages[-1]
            prev_msg = messages[-2] if len(messages) > 1 else None
            
            # Son mesaj ToolMessage ise
            if hasattr(last_msg, 'tool_call_id'):
                tool_call_id = str(last_msg.tool_call_id)
                print(f"[DEBUG] Found tool message with ID: {tool_call_id}")
                
                if "list_tables" in tool_call_id:
                    print("[DEBUG] → get_schema")
                    return "get_schema"
                elif "schema" in tool_call_id:
                    print("[DEBUG] → execute_query")
                    return "execute_query"
                elif "call_" in tool_call_id:  # OpenAI tool call ID
                    print("[DEBUG] → final_answer")
                    return "final_answer"
            
            # Önceki mesajda tool call varsa
            if prev_msg and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                for tool_call in prev_msg.tool_calls:
                    if tool_call.get('name') == "db_query_tool":
                        print("[DEBUG] → final_answer (db_query_tool found)")
                        return "final_answer"
        
        print("[DEBUG] → end")
        return "end"


# SOLID Principle 4: Interface Segregation Principle (ISP)
class ToolManagerInterface(ABC):
    """Tool yönetimi için interface - ISP"""
    
    @abstractmethod
    def get_tools(self) -> List:
        pass


class SQLToolManager(ToolManagerInterface):
    """SQL araçları yönetimi - ISP"""
    
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
    
    def get_tools(self) -> List:
        langgraph_tools = LangGraphSQLTools(self.db, self.llm)
        custom_tools = CustomSQLTools(self.db)
        return langgraph_tools.get_basic_tools() + custom_tools.get_custom_tools()


# SOLID Principle 5: Dependency Inversion Principle (DIP)
class NodeFactory:
    """Node'ları oluşturma sorumluluğu - DIP"""
    
    @staticmethod
    def create_nodes(llm, tools) -> Dict[str, NodeInterface]:
        return {
            "list_tables": ListTablesNode(),
            "get_schema": GetSchemaNode(),
            "execute_query": ExecuteQueryNode(llm, tools),
            "final_answer": FinalAnswerNode()
        }


@register_graph("text2sql")
class Text2SQLGraph(BaseGraph):
    """SOLID prensiplerine uygun Text2SQL Graph - DIP"""
    
    def __init__(self, llm, db):
        super().__init__(llm=llm, state_class=State)
        self.db = db
        
        # Dependency Injection - DIP
        self.tool_manager = SQLToolManager(db, llm.get_chat())
        self.tools = self.tool_manager.get_tools()
        self.nodes = NodeFactory.create_nodes(llm.get_chat(), self.tools)
        self.router = SimpleRouter()
        self.tool_node = ToolNode(self.tools)
    
    def build_graph(self):
        """Graph'ı oluştur - Temiz ve SOLID"""
        memory = MemorySaver()
        graph = StateGraph(State)
        
        # Node'ları ekle
        for node_name, node_instance in self.nodes.items():
            graph.add_node(f"step_{node_name}", node_instance.execute)
        
        graph.add_node("tools", self.tool_node)
        
        # Edge'leri ekle
        graph.add_edge(START, "step_list_tables")
        graph.add_edge("step_list_tables", "tools")
        graph.add_edge("step_get_schema", "tools")
        graph.add_edge("step_execute_query", "tools")
        
        # Routing
        graph.add_conditional_edges(
            "tools",
            self.router.route,
            {
                "get_schema": "step_get_schema",
                "execute_query": "step_execute_query",
                "final_answer": "step_final_answer",
                "end": END
            }
        )
        
        graph.add_edge("step_final_answer", END)
        
        return graph.compile(checkpointer=memory)