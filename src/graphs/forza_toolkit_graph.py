"""
Forza Toolkit Graph
Forza API toolkit'ini kullanan basit graph
"""

import json
from typing import List, Dict, Any
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM, State
from src.tools.forza_api_tools import ForzaAPIToolkit


@register_graph("forza_toolkit")
class ForzaToolkitGraph(BaseGraph):
    """Forza API toolkit'i kullanan graph"""
    
    def __init__(self, llm: LLM):
        super().__init__(llm, State)
        # Forza API toolkit'ini initialize et
        self.forza_toolkit = ForzaAPIToolkit(base_url="http://localhost:8080")
        self.tools = self.forza_toolkit.get_tools()
        
    def process_forza_request_node(self, state: State) -> State:
        """Forza API isteğini işle"""
        try:
            # Kullanıcı girdisini al
            user_input = self._get_user_input(state)
            
            if not user_input:
                return self._add_error_message(state, "Lütfen bir Forza ERP isteği belirtin.")
            
            print(f"Forza isteği işleniyor: {user_input}")
            
            # İsteğin türünü analiz et
            request_type = self._analyze_request_type(user_input)
            state["request_type"] = request_type
            
            # Uygun Forza tool'unu seç ve çalıştır
            result = self._execute_forza_tool(request_type, user_input)
            
            state["forza_result"] = result
            
            return state
            
        except Exception as e:
            return self._add_error_message(state, f"Forza işlem hatası: {str(e)}")
    
    def format_forza_response_node(self, state: State) -> State:
        """Forza yanıtını formatla"""
        try:
            result = state.get("forza_result", "")
            request_type = state.get("request_type", "")
            user_input = state.get("original_query", "")
            
            if not result:
                return self._add_bot_message(state, "Forza ERP'den yanıt alınamadı.")
            
            # Yanıtı formatla
            formatted_response = f"🏢 **Forza ERP Sonucu:**\n\n"
            
            if user_input:
                formatted_response += f"**İstek:** {user_input}\n\n"
            
            formatted_response += f"**Sonuç:**\n{result}\n\n"
            formatted_response += "*Forza ERP API*"
            
            return self._add_bot_message(state, formatted_response)
            
        except Exception as e:
            return self._add_error_message(state, f"Yanıt formatlama hatası: {str(e)}")
    
    def _analyze_request_type(self, user_input: str) -> str:
        """İsteğin türünü analiz et"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['login', 'giriş', 'authenticate', 'oturum']):
            return "login"
        elif any(word in user_lower for word in ['business', 'işletme', 'firma', 'companies']):
            return "businesses"
        elif any(word in user_lower for word in ['branch', 'şube', 'branches', 'şubeler']):
            return "branches"
        elif any(word in user_lower for word in ['my branches', 'benim şubelerim', 'tüm şubeler']):
            return "user_branches"
        else:
            return "general"
    
    def _execute_forza_tool(self, request_type: str, user_input: str) -> str:
        """Uygun Forza tool'unu çalıştır"""
        try:
            # Tool mapping
            tool_map = {
                "login": "login",
                "businesses": "get_businesses_by_user_id", 
                "branches": "get_branches_by_business_id",
                "user_branches": "get_user_branches",
                "general": "get_user_branches"  # Default
            }
            
            tool_name = tool_map.get(request_type, "get_user_branches")
            
            # Tool'u bul ve çalıştır
            for tool in self.tools:
                if tool.name == tool_name:
                    if tool_name == "login":
                        # Login için dummy credentials (gerçek uygulamada user'dan alınacak)
                        return tool._run(username="demo", password="demo")
                    elif tool_name == "get_businesses_by_user_id":
                        # User ID dummy (gerçek uygulamada session'dan alınacak)
                        return tool._run(user_id="1")
                    elif tool_name == "get_branches_by_business_id":
                        # Business ID dummy (gerçek uygulamada user'dan alınacak)
                        return tool._run(business_id="1")
                    elif tool_name == "get_user_branches":
                        # Tam workflow
                        return tool._run(username="demo", password="demo")
                    else:
                        return tool._run("")
            
            return f"'{tool_name}' tool'u bulunamadı."
            
        except Exception as e:
            return f"Tool çalıştırma hatası: {str(e)}"
    
    def _get_user_input(self, state: State) -> str:
        """State'den kullanıcı girdisini al"""
        # Önce user_query'yi kontrol et
        user_input = state.get("user_query", "")
        
        # Yoksa messages'dan son human mesajını al
        if not user_input:
            messages = state.get("messages", [])
            for message in reversed(messages):
                # Message objectı mı kontrol et
                if hasattr(message, 'type') and message.type == 'human':
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            # Content list formatı
                            for content_part in message.content:
                                if isinstance(content_part, dict) and content_part.get('type') == 'text':
                                    user_input = content_part.get('text', '')
                                    if user_input.strip():
                                        break
                        elif isinstance(message.content, str):
                            user_input = message.content
                        break
                # String mesaj formatı (eski format)
                elif isinstance(message, str) and not message.startswith('Bot:'):
                    user_input = message
                    break
        
        # State'e de kaydet
        if user_input:
            state["original_query"] = user_input
        
        return user_input.strip() if user_input else ""
    
    def _add_bot_message(self, state: State, message: str) -> State:
        """Bot mesajını doğru formatta ekle"""
        from langchain.schema import AIMessage
        
        messages = state.get("messages", [])
        bot_message = AIMessage(content=message)
        messages.append(bot_message)
        state["messages"] = messages
        return state
    
    def _add_error_message(self, state: State, error: str) -> State:
        """Hata mesajını ekle"""
        return self._add_bot_message(state, f"❌ Hata: {error}")
    
    def build_graph(self):
        """Graph'ı oluştur"""
        print("Forza Toolkit Graph çalıştırılıyor...")
        
        graph = StateGraph(State)
        
        # Node'ları ekle
        graph.add_node("process_forza_request", RunnableLambda(self.process_forza_request_node))
        graph.add_node("format_forza_response", RunnableLambda(self.format_forza_response_node))
        
        # Edge'leri tanımla
        graph.set_entry_point("process_forza_request")
        graph.add_edge("process_forza_request", "format_forza_response")
        graph.set_finish_point("format_forza_response")
        
        return graph.compile()