# src/graphs/intent_graph.py
"""
Intent Analysis Graph - Kullanıcı intent'ini analiz eder
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from enum import Enum

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.services.app_logger import log
from src.graphs.generic_sql_graph import SQLAgentConfiguration


# Enums
class IntentType(str, Enum):
    COUNT = "count"           # Sayım: "kaç tane", "sayısı"
    SUM = "sum"              # Toplama: "toplam", "total tutar"
    LIST = "list"            # Listeleme: "göster", "hangileri"
    GROUP = "group"          # Gruplama: "kategoriye göre", "dağılım"
    FILTER = "filter"        # Filtreleme: "şartını sağlayan"
    COMPARE = "compare"      # Karşılaştırma: "farkı", "hangisi daha"
    TREND = "trend"          # Trend analizi: "seyre git", "artış/azalış"


class ConversationType(str, Enum):
    NEW_QUERY = "new_query"                    # Sıfırdan yeni soru
    FOLLOW_UP = "follow_up"                    # Önceki soruyla ilgili ek soru
    DRILL_DOWN = "drill_down"                  # Önceki sonuçta detaya inme
    REFINEMENT = "refinement"                  # Önceki sorguyu daraltma/genişletme
    CLARIFICATION = "clarification"            # Belirsizlik giderme
    COMPARISON_EXTENSION = "comparison_extension"  # Önceki sonuca karşılaştırma ekleme


# State schema
class IntentAnalysisState(TypedDict):
    user_message: Optional[str]
    table_info: Optional[Dict[str, Any]]
    intent_analysis: Optional[Dict[str, Any]]
    analysis_result: Optional[str]
    errors: Optional[Dict[str, Any]]
    messages: Optional[List[Any]]


# Intent Analysis Model
class UserIntentAnalysisResult(BaseModel):
    """Intent analiz sonuçları"""
    intent_type: IntentType = Field(description="Kullanıcının temel SQL intent'i")
    conversation_type: ConversationType = Field(description="Konuşma akışındaki pozisyon")
    confidence: float = Field(ge=0.0, le=1.0, description="Intent classification güvenilirlik skoru")
    target_action: str = Field(description="Kullanıcının yapmaya çalıştığı işlemin açıklaması")
    reasoning: Optional[str] = Field(default=None, description="Intent classification gerekçesi")
    requires_context: bool = Field(default=False, description="Önceki konuşma context'i gerekli mi?")


@register_graph("intent_analysis")
class IntentAnalysisGraph(BaseGraph):
    """Intent Analysis Graph - kullanıcı intent analizi"""
    
    def __init__(self, llm, memory: MemorySaver = None):
        super().__init__(llm=llm, state_class=IntentAnalysisState, memory=memory)
        self.logger = log.get(module="intent_analysis_graph")
        self.analyze_intent_tool = self._create_analyze_intent_tool()
        
        self.logger.info("Intent Analysis Graph initialized")
    
    def _create_analyze_intent_tool(self):
        """Intent analiz tool'unu oluştur"""
        
        # Outer scope'dan referansları al
        llm = self.llm
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
    
    def build_graph(self):
        """Build Intent Analysis graph"""
        self.logger.info("Building Intent Analysis graph")
        
        try:
            if self.memory is None:
                self.memory = MemorySaver()
            
            graph_builder = StateGraph(IntentAnalysisState, config_schema=SQLAgentConfiguration)
            
            # 1. Analyze Intent Node
            def analyze_intent_node(state: IntentAnalysisState, *, config) -> IntentAnalysisState:
                """Intent analiz node'u - tool çağırır ve ToolMessage ekler"""
                try:
                    configurable = config.get("configurable", {})
                    
                    # State'den bilgileri al
                    user_message = state.get("user_message", "")
                    table_info = state.get("table_info", {})
                    
                    if not user_message:
                        error_msg = "User message not found in state"
                        state["errors"] = {"error_message": error_msg}
                        state["messages"] = [ToolMessage(content=error_msg, tool_call_id="analyze_intent")]
                        return state
                    
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
            
            # Add node
            graph_builder.add_node("analyze_intent", analyze_intent_node)
            
            # Define edges
            graph_builder.set_entry_point("analyze_intent")
            graph_builder.set_finish_point("analyze_intent")
            
            # Compile graph
            compiled_graph = graph_builder.compile(
                checkpointer=self.memory,
                name="intent_analysis_graph"
            )
            
            self.logger.info("Intent Analysis graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build Intent Analysis graph", error=str(e))
            raise
    
    def invoke(self, config: Dict = None) -> Dict:
        """Main entry point for intent analysis"""
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
                config["configurable"]["thread_id"] = "intent_analysis_session"
            
            # Create initial state
            initial_state = {}
            
            # Execute workflow
            result_state = self._compiled_graph.invoke(initial_state, config)
            
            # Return result
            return {
                "success": "errors" not in result_state or not result_state.get("errors", {}).get("error_message"),
                "intent_analysis": result_state.get("intent_analysis", {}),
                "message": result_state.get("analysis_result", "İşlem tamamlandı."),
                "errors": result_state.get("errors", {})
            }
            
        except Exception as e:
            self.logger.error("Intent Analysis workflow failed", error=str(e))
            return {
                "success": False,
                "intent_analysis": {},
                "message": f"Sistem hatası: {str(e)}",
                "errors": {"error_message": str(e)}
            }