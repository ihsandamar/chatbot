# src/graphs/erp_main_graph.py
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Union
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Discriminator, Field, Tag
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.graphs.text2sql_graph import Text2SQLGraph
from src.models.models import LLM
from src.services.app_logger import log
from enum import Enum
from langchain_core.messages.base import Serializable
from langchain_core.messages.utils import _get_type

# %%
################ Define State Schemas ################

class ModuleType(str, Enum):
    """ERP Modül türleri"""
    REPORTING = "reporting"
    TEXT2SQL = "text2sql" 
    CUSTOMER_SERVICE = "customer_service"
    SUPPORT = "support"
    GENERAL = "general"

class ModuleResult(TypedDict):
    """Modül sonucu için state"""
    module_name: str
    result: Dict[str, Any]
    processing_time: float
    success: bool


class ERPChatbotState(TypedDict):   
    """Ana ERP Chatbot State - Sadece dropdown"""
    
    # dropdown seçenekleri
    module_choosing: ModuleType

# %%
################ Define Node Functions ################

@register_graph("erp_main")
class ERPMainGraph(BaseGraph):
    """ERP Ana Graph - Basit yapı"""
    
    def __init__(self, llm: LLM):
        super().__init__(llm=llm, state_class=ERPChatbotState)
        self.logger = log.get(module="erp_main_graph")
    
    def routing_node(self, state: ERPChatbotState) -> ERPChatbotState:
        """Seçilen modülü çalıştıran routing node'u"""
        self.logger.info("=== ROUTING START ===")
        
        selected_module = state.get('module_choosing', ModuleType.TEXT2SQL)
        self.logger.info("Selected module", module=selected_module)
        
        if selected_module == ModuleType.TEXT2SQL:
            # Text2SQL graphını çalıştır
            
            text2sql_graph = Text2SQLGraph(llm=self.llm).build_graph()
            
            # Örnek mesaj ile text2sql graph'ı çağır
            result = text2sql_graph.invoke({
                "messages": [HumanMessage(content="Müşteriler tablosunu listele")]
            })
            
            self.logger.info("Text2SQL graph executed successfully")
            
            return {
                'module_choosing': selected_module
            }
        
        return {
            'module_choosing': selected_module
        }
    
    # %%
    ################ Router Functions ################
    
    # %%
    ################ Build Graph ################
    
    def build_graph(self):
        """Graf yapısını oluştur - Sadece routing"""
        self.logger.info("Building ERP main graph with simple routing structure")
        
        memory = MemorySaver()
        graph = StateGraph(ERPChatbotState)
        
        # Sadece routing node'u ekle
        graph.add_node("routing", self.routing_node)
        
        # Basit yapı: START -> routing -> END
        graph.add_edge(START, "routing")
        graph.add_edge("routing", END)
        
        # Compile
        compiled_graph = graph.compile(
            name="erp_main_graph",
            checkpointer=memory
        )
        
        self.logger.info("ERP main graph compiled successfully")
        return compiled_graph

# %%
################ Test Function ################

def test_erp_graph():
    """ERP Graph'ı test et"""
    from src.models.models import LLM
    from src.services.config_loader import ConfigLoader
    
    print("=== ERP CHATBOT GRAPH TEST ===")
    
    # Setup
    config = ConfigLoader.load_config()
    llm = LLM(model="gpt-4o-mini", temperature=0.0, api_key=config.llm.api_key)
    
    # Graph oluştur
    erp_graph = ERPMainGraph(llm).build_graph()
    
    # Test cases
    test_cases = [
        "Merhaba",
        "Müşteriler tablosunu göster",
        "Bu ayın ciro raporunu hazırla",
        "Sipariş durumumu sorgula"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {test_input}")
        print('='*50)
        
        initial_state = {'messages': [HumanMessage(content=test_input)]}
        config = {"configurable": {"thread_id": f"test_{i}"}}
        
        # Debug mode ile çalıştır
        try:
            result = erp_graph.invoke(initial_state, config=config, debug=True)
            
            print("\n=== FINAL STATE ===")
            print(f"Intent: {result.get('current_intent', 'N/A')}")
            print(f"Target Module: {result.get('target_module', 'N/A')}")
            print(f"Processing Complete: {result.get('processing_complete', False)}")
            print(f"Module Results Count: {len(result.get('module_results', []))}")
            print(f"Messages Count: {len(result.get('messages', []))}")
            
            if result.get('debug_info'):
                print(f"Debug Info: {result['debug_info']}")
                
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    test_erp_graph()