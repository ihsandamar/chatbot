# src/graphs/query_extraction_graph.py
"""
Query Extraction Graph - SQL tablo şemasını çeker ve distinct değerleri toplar
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.services.app_logger import log
from src.graphs.generic_sql_graph import SQLAgentConfiguration
from typing import Optional, TypedDict, Annotated, List, Any
from langgraph.graph.message import add_messages, AnyMessage



# Define the state schema 
class Text2SQLState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    table_info: Optional[Dict[str, Any]]
    extraction_result: Optional[str]
    errors: Optional[Dict[str, Any]]

    #intent analysis state
    intent_analysis: Optional[Dict[str, Any]]
    analysis_result: Optional[str]



class Table(BaseModel):
    """Table in SQL database with detailed schema information."""
    name: str = Field(description="Name of table in SQL database.")
    columns: List[Dict[str, Any]] = Field(default=[], description="Column definitions with distinct values")
    relations: List[str] = Field(default=[], description="Table relations")


@register_graph("query_extraction")
class QueryExtractionGraph(BaseGraph):
    """Query Extraction Graph - tablo şeması ve distinct değerler"""
    
    def __init__(self, llm, db: SQLDatabase = None, memory: MemorySaver = None):
        super().__init__(llm=llm, state_class=Text2SQLState, memory=memory)
        self.logger = log.get(module="query_extraction_graph")
        self.db = db
        self.prepare_table_data_tool = self._create_prepare_table_data_tool()
        self.analyze_intent_tool = self._create_analyze_intent_tool()
        
        self.logger.info("Query Extraction Graph initialized")
    
    def _is_string_column(self, column_type: str) -> bool:
        """Kolonun string tipinde olup olmadığını kontrol et"""
        string_types = ['varchar', 'text', 'char', 'string', 'nvarchar', 'ntext']
        return any(str_type in column_type.lower() for str_type in string_types)
    

    def _create_analyze_intent_tool(self):
        """Intent analiz tool'unu oluştur"""
        
        # Outer scope'dan referansları al
        llm = self.llm.get_chat()
        logger = self.logger
        
        @tool
        def analyze_intent_tool(user_message: str, table_info: Dict = None) -> Dict:
            """Kullanıcı mesajını analiz eder ve intent'i belirler"""
            try:
                # Tablo bilgileri varsa context'e ekle
                table_context = ""
                if table_info and table_info.get("columns"):
                    columns_info = [col.get("name", "") for col in table_info.get("columns", [])]
                    table_context = f"\\nMevcut Tablo Kolonları: {', '.join(columns_info)}"
                
                # LLM Prompt
                prompt = f"""Kullanıcının mesajını analiz et ve SQL intent'ini belirle:

Kullanıcı Mesajı: "{user_message}"{table_context}

Intent Türleri:
- count: Sayım işlemleri ("kaç tane", "sayısı")
- sum: Toplama işlemleri ("toplam", "total tutar")
- list: Listeleme işlemleri ("göster", "hangileri")
- group: Gruplama işlemleri ("kategoriye göre", "dağılım")
- filter: Filtreleme işlemleri ("şartını sağlayan")
- compare: Karşılaştırma işlemleri ("farkı", "hangisi daha")
- trend: Trend analizi ("seyre git", "artış/azalış")

Konuşma Türleri:
- new_query: Sıfırdan yeni soru
- follow_up: Önceki soruyla ilgili ek soru  
- drill_down: Önceki sonuçta detaya inme
- refinement: Önceki sorguyu daraltma/genişletme
- clarification: Belirsizlik giderme
- comparison_extension: Önceki sonuca karşılaştırma ekleme

JSON formatında döndür:
{{
    "intent_type": "count|sum|list|group|filter|compare|trend",
    "conversation_type": "new_query|follow_up|drill_down|refinement|clarification|comparison_extension",
    "confidence": 0.0-1.0,
    "target_action": "kısa açıklama - ne yapmaya çalışıyor",
    "reasoning": "neden bu intent'i seçtim",
    "requires_context": true/false
}}"""
                
                # LLM ile analiz yap
                response = llm.invoke([{"role": "user", "content": prompt}])
                
                # JSON parse et
                import json
                analysis = json.loads(response.content)
                
                logger.info("Intent analysis completed", intent=analysis.get("intent_type"))
                return analysis
                
            except Exception as e:
                logger.error("Intent analysis failed", error=str(e))
                # Hata durumunda varsayılan analiz döndür
                return {
                    "intent_type": "list",
                    "conversation_type": "new_query", 
                    "confidence": 0.5,
                    "target_action": f"Kullanıcı sorgu yapmaya çalışıyor: {user_message[:50]}...",
                    "reasoning": f"LLM analizi başarısız: {str(e)}",
                    "requires_context": False
                }
        
        return analyze_intent_tool
    

    def _create_prepare_table_data_tool(self):
        """prepare_table_data tool'unu oluştur"""
        
        # Outer scope'dan self referanslarını alalım
        db = self.db
        logger = self.logger
        is_string_column = self._is_string_column
        
        @tool
        def prepare_table_data(table_name: str, table_columns: List[Dict], max_distinct_values: int = 50) -> Dict:
            """Config'ten aldığı bilgiler ile table_info'yu doldurur"""
            try:
                execute_tool = QuerySQLDatabaseTool(db=db)
                enriched_columns = []
                
                for config_col in table_columns:
                    col_name = config_col.get("name", "") if isinstance(config_col, dict) else str(config_col)
                    col_type = config_col.get("type", "") if isinstance(config_col, dict) else ""
                    col_description = config_col.get("description", "") if isinstance(config_col, dict) else ""
                    
                    column_info = {
                        "name": col_name,
                        "type": col_type,
                        "description": col_description,
                        "distinct_values": []
                    }
                    
                    # String kolonlar için distinct values çek
                    if col_name and is_string_column(col_type):
                        try:
                            # Get distinct values
                            query = f"SELECT DISTINCT TOP {max_distinct_values} [{col_name}] FROM [{table_name}] WHERE [{col_name}] IS NOT NULL"
                            result = execute_tool.invoke(query)
                            
                            # Parse result
                            values = []
                            for line in result.split('\n'):
                                line = line.strip()
                                if line and not line.startswith('-') and not line.startswith('('):
                                    values.append(line)
                            
                            column_info["distinct_values"] = values[:max_distinct_values]
                            logger.info(f"Collected {len(values)} distinct values for {col_name}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to get distinct values for {col_name}", error=str(e))
                            column_info["distinct_values"] = []
                    
                    enriched_columns.append(column_info)
                
                # Table model oluştur
                table_info = Table(
                    name=table_name,
                    columns=enriched_columns,
                    relations=[]
                )
                
                # Table info'yu döndür
                return {
                    "table_name": table_info.name,
                    "columns": table_info.columns,
                    "relations": table_info.relations
                }
                
            except Exception as e:
                logger.error("Table data preparation failed", error=str(e))
                raise Exception(f"Table data preparation failed: {str(e)}")
        
        return prepare_table_data
    



    
    
    def build_graph(self):
        """Build Query Extraction graph"""
        self.logger.info("Building Query Extraction graph")
        
        try:
            if self.memory is None:
                self.memory = MemorySaver()
            
            graph_builder = StateGraph(Text2SQLState, config_schema=SQLAgentConfiguration)
            
            # 1. Prepare Table Data Node  
            def prepare_table_data_node(state: Text2SQLState, *, config) -> Text2SQLState:
                """Tool'u çağırır ve ToolMessage ekler"""
                try:
                    configurable = config.get("configurable", {})
                    
                    # Config'ten table bilgilerini al
                    table_name = configurable.get("table_name", "")
                    config_columns = configurable.get("table_columns", [])
                    max_distinct_values = configurable.get("max_distinct_values", 50)
                    
                    if not table_name or not config_columns:
                        error_msg = "Table name or columns not found in config"
                        state["errors"] = {"error_message": error_msg}
                        state["messages"] = [ToolMessage(content=error_msg, tool_call_id="prepare_table_data")]
                        return state
                    
                    # Tool'u çağır
                    tool_result = self.prepare_table_data_tool.invoke({
                        "table_name": table_name,
                        "table_columns": config_columns,
                        "max_distinct_values": max_distinct_values
                    })
                    
                    # State'e kaydet
                    state["table_info"] = tool_result
                    
                    # Success message
                    columns = tool_result.get("columns", [])
                    searchable_columns = [col["name"] for col in columns if col.get("distinct_values")]
                    success_msg = f"""Tablo şeması başarıyla çekildi:

• **Tablo:** {table_name}
• **Kolonlar:** {len(columns)} adet
• **Aranabilir Kolonlar:** {len(searchable_columns)} adet

**Distinct değerler çekilen kolonlar:**
{', '.join(searchable_columns)}"""
                    
                    state["extraction_result"] = success_msg
                    state["messages"] = [ToolMessage(content=success_msg, tool_call_id="prepare_table_data")]
                    
                    self.logger.info("Table schema extracted successfully", 
                                   table=table_name, 
                                   columns=len(columns),
                                   searchable=len(searchable_columns))
                    return state
                    
                except Exception as e:
                    error_msg = f"Tablo şeması çekilirken hata oluştu: {str(e)}"
                    state["errors"] = {"error_message": error_msg}
                    state["extraction_result"] = error_msg
                    state["messages"] = [ToolMessage(content=error_msg, tool_call_id="prepare_table_data")]
                    
                    self.logger.error("Table schema extraction failed", error=str(e))
                    return state
            
            # 2. Analyze Intent Node
            def analyze_intent_node(state: Text2SQLState, *, config) -> Text2SQLState:
                """Intent analiz node'u - tool çağırır ve ToolMessage ekler"""
                try:
                    configurable = config.get("configurable", {})
                    
                    # Messages'den son user mesajını al
                    messages = state.get("messages", [])
                    user_message = ""
                    
                    # Son human/user mesajını bul
                    for msg in reversed(messages):
                        if hasattr(msg, 'type') and msg.type == "human":
                            user_message = msg.content
                            break
                        elif hasattr(msg, 'role') and msg.role == "user":
                            user_message = msg.content
                            break
                    
                    if not user_message:
                        error_msg = "No user message found in messages"
                        state["errors"] = {"error_message": error_msg}
                        state["messages"] = [ToolMessage(content=error_msg, tool_call_id="analyze_intent")]
                        return state
                    
                    table_info = state.get("table_info", {})
                    
                    # Tool'u çağır
                    tool_result = self.analyze_intent_tool.invoke({
                        "user_message": user_message,
                        "table_info": table_info
                    })
                    
                    # State'e kaydet
                    state["intent_analysis"] = tool_result
                    
                    # Success message
                    success_msg = f"""Intent analizi tamamlandı:

• **Intent Türü:** {tool_result.get('intent_type', 'N/A')}
• **Konuşma Türü:** {tool_result.get('conversation_type', 'N/A')}
• **Güven Skoru:** {tool_result.get('confidence', 0):.2f}
• **Hedef Aksiyon:** {tool_result.get('target_action', 'N/A')}

**Analiz Gerekçesi:** {tool_result.get('reasoning', 'N/A')}"""
                    
                    state["analysis_result"] = success_msg
                    state["messages"] = [ToolMessage(content=success_msg, tool_call_id="analyze_intent")]
                    
                    self.logger.info("Intent analysis completed successfully", 
                                   intent=tool_result.get("intent_type"),
                                   confidence=tool_result.get("confidence"))
                    return state
                    
                except Exception as e:
                    error_msg = f"Intent analizi sırasında hata oluştu: {str(e)}"
                    state["errors"] = {"error_message": error_msg}
                    state["analysis_result"] = error_msg
                    state["messages"] = [ToolMessage(content=error_msg, tool_call_id="analyze_intent")]
                    
                    self.logger.error("Intent analysis failed", error=str(e))
                    return state
            
            # Add nodes
            graph_builder.add_node("prepare_table_data", prepare_table_data_node)
            graph_builder.add_node("analyze_intent", analyze_intent_node)
            
            # Define edges
            graph_builder.set_entry_point("prepare_table_data")
            graph_builder.add_edge("prepare_table_data", "analyze_intent")
            graph_builder.set_finish_point("analyze_intent")
            
            # Compile graph
            compiled_graph = graph_builder.compile(
                checkpointer=self.memory,
                name="query_extraction_graph"
            )
            
            self.logger.info("Query Extraction graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build Query Extraction graph", error=str(e))
            raise
    
    def invoke(self, config: Dict = None) -> Dict:
        """Main entry point for query extraction"""
        try:
            # Build graph if needed
            if not hasattr(self, '_compiled_graph'):
                self._compiled_graph = self.build_graph()
            
            # Use thread_id for state persistence
            if config is None:
                config = {}
            
            # Ensure thread_id exists for state persistence
            if "configurable" not in config:
                config["configurable"] = {}
            if "thread_id" not in config["configurable"]:
                config["configurable"]["thread_id"] = "extraction_session"
            
            # Create initial state
            initial_state = {}
            
            # Execute workflow
            result_state = self._compiled_graph.invoke(initial_state, config)
            
            # Return result
            return {
                "success": "errors" not in result_state or not result_state.get("errors", {}).get("error_message"),
                "table_info": result_state.get("table_info", {}),
                "message": result_state.get("extraction_result", "İşlem tamamlandı."),
                "errors": result_state.get("errors", {})
            }
            
        except Exception as e:
            self.logger.error("Query Extraction workflow failed", error=str(e))
            return {
                "success": False,
                "table_info": {},
                "message": f"Sistem hatası: {str(e)}",
                "errors": {"error_message": str(e)}
            }