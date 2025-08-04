# src/graphs/supervisor_graph.py
"""
ERP Chatbot Supervisor Graph
LangGraph supervisor pattern ile routing ve state management
"""

from typing import Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM
from src.models.erp_langgraph_models import (
    ERPChatbotState, ModuleType, UserInfo, 
    create_empty_erp_state, create_sample_user_info,
    ChatbotResponseData, ButtonAction
)
from src.services.app_logger import log

# ==============================================================================
# SUPERVISOR ROUTING MODEL
# ==============================================================================

class SupervisorRouting(BaseModel):
    """Supervisor routing kararları için Pydantic model"""
    next_action: Literal[
        "welcome_user",
        "reporting_module", 
        "support_module",
        "documents_training_module",
        "request_module", 
        "company_info_module",
        "other_module",
        "clarify_request",
        "end_conversation"
    ]
    confidence: float
    reasoning: str
    detected_intent: str
    required_module: str

# ==============================================================================
# SUPERVISOR GRAPH IMPLEMENTATION
# ==============================================================================

@register_graph("supervisor")
class SupervisorGraph(BaseGraph):
    """
    ERP Chatbot Supervisor Graph
    - User intent detection
    - Module routing decisions
    - State management
    - Welcome flow coordination
    """
    
    def __init__(self, llm: LLM):
        self._logger = log.get(module="graphs", file="supervisor_graph", cls="SupervisorGraph")
        self._logger.debug("Initializing SupervisorGraph")
        super().__init__(llm=llm, state_class=ERPChatbotState)
        
        # Supervisor routing model with tools
        self.supervisor_model = self._create_supervisor_model()
        
    def _create_supervisor_model(self):
        """Supervisor için LLM model oluştur"""
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_supervisor_system_prompt()),
            ("placeholder", "{messages}")
        ])
        
        return supervisor_prompt | self.llm.get_chat().bind_tools(
            [SupervisorRouting], 
            tool_choice="SupervisorRouting"
        )
    
    def _get_supervisor_system_prompt(self) -> str:
        """Supervisor sistem prompt'u"""
        return """Sen ForzaChatBot'un merkezi koordinatörüsün. ERP müşteri hizmetleri chatbotu olarak çalışıyorsun.

🎯 **Ana Görevin:**
Kullanıcı mesajlarını analiz ederek hangi modüle yönlendirileceğini belirlemek ve state'i güncellemek.

📋 **Mevcut Modüller:**
1. **REPORTING** - Raporlama işlemleri (ciro, satış, analiz, grafik)
2. **SUPPORT** - Teknik destek (kurulum, hata, yardım)  
3. **DOCUMENTS_TRAINING** - Dokümanlar ve eğitim materyalleri
4. **REQUEST** - Yeni özellik talepleri ve istekler
5. **COMPANY_INFO** - Firma hakkında bilgiler
6. **OTHER** - Genel sorular, dış API'ler (döviz, hava durumu)

🔍 **Intent Detection Kuralları:**

**REPORTING** anahtar kelimeler:
- "rapor", "report", "ciro", "satış", "analiz", "grafik", "chart"
- "master rapor", "muhasebe", "dinamik rapor"
- "ne kadar", "göster", "listele", "özet", "istatistik"
- Şube/tarih bazlı sorular

**SUPPORT** anahtar kelimeler:  
- "destek", "yardım", "help", "hata", "error", "problem"
- "kurulum", "çalışmıyor", "nasıl", "how to"

**REQUEST** anahtar kelimeler:
- "istek", "talep", "özellik", "geliştirme", "ekle"
- "yeni", "request", "feature"

**COMPANY_INFO** anahtar kelimeler:
- "firma", "şirket", "hakkında", "about", "company"
- "iletişim", "adres", "telefon"

**OTHER** anahtar kelimeler:
- "döviz", "kur", "hava durumu", "weather"
- Genel sorular, tanımlanamayan istekler

📊 **Routing Logic:**
1. İlk mesajda kullanıcıyı karşıla → "welcome_user"
2. Intent confidence > 0.8 → Direct routing
3. Intent confidence < 0.8 → "clarify_request"  
4. Anlaşılmayan → "clarify_request"
5. Çıkış belirtilerse → "end_conversation"

🎭 **Kişilik:**
- Profesyonel ama samimi
- Türkçe konuş
- Kısa ve net cevaplar ver
- ERP terminolojisini bil

Kullanıcı mesajını analiz et ve SupervisorRouting tool'u ile routing kararını ver."""

    # ========================================================================
    # NODE IMPLEMENTATIONS
    # ========================================================================
    
    def welcome_node(self, state: ERPChatbotState) -> ERPChatbotState:
        """Kullanıcıyı karşılama node'u"""
        logger = self._logger.bind(node="welcome_node")
        logger.debug("Welcome node executing")
        
        # Sample user info oluştur (gerçek uygulamada authentication'dan gelir)
        user_info = create_sample_user_info()
        
        # Welcome mesajı oluştur
        welcome_message = self._create_welcome_message(user_info)
        
        # State güncelle
        updated_state = state.copy()
        updated_state["user_info"] = user_info
        updated_state["workflow_step"] = "welcomed"
        updated_state["session_context"] = {
            "welcomed_at": "2025-08-02T10:00:00Z",
            "session_type": "new_session"
        }
        updated_state["messages"] = state["messages"] + [welcome_message]
        
        logger.info("User welcomed", user_id=user_info["user_id"])
        return updated_state
    
    def intent_detection_node(self, state: ERPChatbotState) -> ERPChatbotState:
        """Intent detection ve routing kararı node'u"""
        logger = self._logger.bind(node="intent_detection_node")
        logger.debug("Intent detection executing")
        
        # Son kullanıcı mesajını al
        user_message = self._get_last_user_message(state["messages"])
        logger.debug("Analyzing user message", message_preview=user_message[:100])
        
        # Supervisor model ile intent detection
        try:
            response = self.supervisor_model.invoke({"messages": state["messages"]})
            
            # Tool call'dan routing kararını çıkar
            if response.tool_calls:
                routing_decision = response.tool_calls[0]["args"]
                logger.info("Routing decision made", 
                           next_action=routing_decision["next_action"],
                           confidence=routing_decision["confidence"])
                
                # State güncelle
                updated_state = state.copy()
                updated_state["workflow_step"] = "intent_detected"
                updated_state["conversation_metadata"] = {
                    "routing_decision": routing_decision,
                    "detected_at": "2025-08-02T10:01:00Z"
                }
                
                # Modül seçimini state'e ekle
                module_mapping = {
                    "reporting_module": ModuleType.REPORTING,
                    "support_module": ModuleType.SUPPORT,
                    "documents_training_module": ModuleType.DOCUMENTS_TRAINING,
                    "request_module": ModuleType.REQUEST,
                    "company_info_module": ModuleType.COMPANY_INFO,
                    "other_module": ModuleType.OTHER
                }
                
                if routing_decision["next_action"] in module_mapping:
                    updated_state["current_module"] = module_mapping[routing_decision["next_action"]]
                    logger.debug("Module selected", module=updated_state["current_module"])
                
                # Routing response mesajı ekle
                routing_message = self._create_routing_response(routing_decision)
                updated_state["messages"] = state["messages"] + [routing_message]
                
                return updated_state
                
        except Exception as e:
            logger.error("Intent detection failed", error=str(e))
            # Fallback routing
            return self._create_fallback_response(state)
    
    def clarification_node(self, state: ERPChatbotState) -> ERPChatbotState:
        """Netleştirme sorusu node'u"""
        logger = self._logger.bind(node="clarification_node")
        logger.debug("Clarification node executing")
        
        clarification_message = AIMessage(
            content="""🤔 **Hangi konuda yardımcı olabilirim?**

Size aşağıdaki konularda yardımcı olabilirim:

🔹 **Raporlama** - Ciro, satış, analiz raporları ve grafikler
🔹 **Destek** - Teknik yardım, kurulum, hata çözümü  
🔹 **Dokümanlar** - Kullanım kılavuzları ve eğitim materyalleri
🔹 **Talep** - Yeni özellik istekleri
🔹 **Firma Bilgisi** - Şirketimiz hakkında bilgiler
🔹 **Diğer** - Genel sorular

Lütfen hangi konuda yardım istediğinizi belirtin."""
        )
        
        updated_state = state.copy()
        updated_state["workflow_step"] = "clarification_requested"
        updated_state["requires_user_input"] = True
        updated_state["messages"] = state["messages"] + [clarification_message]
        
        logger.info("Clarification requested")
        return updated_state
    
    def routing_decision_node(self, state: ERPChatbotState) -> ERPChatbotState:
        """Final routing kararı node'u"""
        logger = self._logger.bind(node="routing_decision_node")
        logger.debug("Routing decision node executing")
        
        # Routing metadata'sından karar al
        routing_metadata = state.get("conversation_metadata", {}).get("routing_decision", {})
        next_action = routing_metadata.get("next_action", "clarify_request")
        
        logger.info("Final routing decision", next_action=next_action)
        
        # State'i route edilmeye hazır hale getir
        updated_state = state.copy()
        updated_state["workflow_step"] = f"routed_to_{next_action}"
        
        # Routing confirmation mesajı
        routing_confirmation = self._create_routing_confirmation(next_action)
        updated_state["messages"] = state["messages"] + [routing_confirmation]
        
        return updated_state
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_last_user_message(self, messages: List) -> str:
        """Son kullanıcı mesajını al"""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list) and len(msg.content) > 0:
                        return msg.content[0].get('text', '')
                    elif isinstance(msg.content, str):
                        return msg.content
        return ""
    
    def _create_welcome_message(self, user_info: UserInfo) -> AIMessage:
        """Karşılama mesajı oluştur"""
        return AIMessage(
            content=f"""👋 **Merhaba {user_info['username']}!**

Ben **ForzaChatBot**, ERP müşteri hizmetleri asistanınız. Size aşağıdaki konularda yardımcı olabilirim:

🔹 **[Raporlama]** - Ciro, satış ve analiz raporları
🔹 **[Destek]** - Teknik yardım ve sorun çözümü  
🔹 **[Dokümanlar & Eğitim]** - Kullanım kılavuzları
🔹 **[Talep]** - Yeni özellik istekleri
🔹 **[Firma Bilgisi]** - Şirket hakkında bilgiler
🔹 **[Diğer]** - Genel sorular

Nasıl yardımcı olabilirim? 😊"""
        )
    
    def _create_routing_response(self, routing_decision: Dict[str, Any]) -> AIMessage:
        """Routing response mesajı oluştur"""
        module_messages = {
            "reporting_module": "📊 **Raporlama** modülüne yönlendiriyorum...",
            "support_module": "🛠️ **Destek** modülüne yönlendiriyorum...",
            "documents_training_module": "📚 **Dokümanlar** modülüne yönlendiriyorum...",
            "request_module": "💡 **Talep** modülüne yönlendiriyorum...",
            "company_info_module": "🏢 **Firma Bilgisi** modülüne yönlendiriyorum...",
            "other_module": "🔍 **Genel** modülüne yönlendiriyorum...",
            "clarify_request": "🤔 Talebinizi netleştirmek için birkaç soru sormam gerekiyor...",
            "end_conversation": "👋 Görüşmek üzere! İyi günler dilerim."
        }
        
        message = module_messages.get(
            routing_decision["next_action"], 
            "🔄 İsteğinizi işleme alıyorum..."
        )
        
        return AIMessage(content=message)
    
    def _create_routing_confirmation(self, next_action: str) -> AIMessage:
        """Routing onay mesajı oluştur"""
        return AIMessage(
            content=f"✅ **{next_action}** modülüne başarıyla yönlendirildiniz.\n\n" +
                   "İlgili uzman ekibimiz tarafından size yardımcı olunacak."
        )
    
    def _create_fallback_response(self, state: ERPChatbotState) -> ERPChatbotState:
        """Fallback response oluştur"""
        logger = self._logger.bind(node="fallback")
        logger.warning("Creating fallback response")
        
        fallback_message = AIMessage(
            content="⚠️ Maalesef isteğinizi tam olarak anlayamadım. " +
                   "Lütfen hangi konuda yardım istediğinizi daha detayına açıklayabilir misiniz?"
        )
        
        updated_state = state.copy()
        updated_state["workflow_step"] = "fallback_response"
        updated_state["requires_user_input"] = True
        updated_state["messages"] = state["messages"] + [fallback_message]
        
        return updated_state
    
    # ========================================================================
    # ROUTING LOGIC
    # ========================================================================
    
    def should_continue(self, state: ERPChatbotState) -> Literal["welcome", "intent_detection", "clarification", "routing_decision", "end"]:
        """Routing logic - hangi node'a gidileceğini belirle"""
        logger = self._logger.bind(node="routing_logic")
        
        workflow_step = state.get("workflow_step", "initial")
        messages = state.get("messages", [])
        user_info = state.get("user_info")
        
        logger.debug("Routing decision", workflow_step=workflow_step, message_count=len(messages))
        
        # İlk mesaj mı kontrol et
        if not user_info and len(messages) <= 1:
            logger.debug("Routing to welcome")
            return "welcome"
        
        # Workflow step'e göre routing
        if workflow_step == "welcomed":
            logger.debug("Routing to intent_detection")
            return "intent_detection"
        
        if workflow_step == "intent_detected":
            # Routing decision metadata'sını kontrol et
            routing_metadata = state.get("conversation_metadata", {}).get("routing_decision", {})
            confidence = routing_metadata.get("confidence", 0.0)
            next_action = routing_metadata.get("next_action", "")
            
            if confidence < 0.8 or next_action == "clarify_request":
                logger.debug("Routing to clarification")
                return "clarification"
            else:
                logger.debug("Routing to routing_decision")
                return "routing_decision"
        
        if workflow_step == "clarification_requested":
            logger.debug("Routing to intent_detection after clarification")
            return "intent_detection"
        
        if workflow_step.startswith("routed_to_"):
            logger.debug("Routing to end")
            return "end"
        
        # Default fallback
        logger.debug("Default routing to intent_detection")
        return "intent_detection"
    
    # ========================================================================
    # GRAPH BUILDER
    # ========================================================================
    
    def build_graph(self):
        """Supervisor graph'ı oluştur"""
        logger = self._logger.bind(method="build_graph")
        logger.debug("Building supervisor graph")
        
        memory = MemorySaver()
        graph = StateGraph(ERPChatbotState)
        
        # Node'ları ekle
        graph.add_node("welcome", self.welcome_node)
        graph.add_node("intent_detection", self.intent_detection_node)
        graph.add_node("clarification", self.clarification_node)
        graph.add_node("routing_decision", self.routing_decision_node)
        
        logger.debug("Nodes added to graph")
        
        # Entry point
        graph.add_edge(START, "welcome")
        
        # Conditional routing
        graph.add_conditional_edges(
            "welcome",
            self.should_continue,
            {
                "welcome": "welcome",
                "intent_detection": "intent_detection",
                "clarification": "clarification", 
                "routing_decision": "routing_decision",
                "end": END
            }
        )
        
        graph.add_conditional_edges(
            "intent_detection",
            self.should_continue,
            {
                "welcome": "welcome",
                "intent_detection": "intent_detection",
                "clarification": "clarification",
                "routing_decision": "routing_decision", 
                "end": END
            }
        )
        
        graph.add_conditional_edges(
            "clarification", 
            self.should_continue,
            {
                "welcome": "welcome",
                "intent_detection": "intent_detection",
                "clarification": "clarification",
                "routing_decision": "routing_decision",
                "end": END
            }
        )
        
        graph.add_conditional_edges(
            "routing_decision",
            self.should_continue,
            {
                "welcome": "welcome", 
                "intent_detection": "intent_detection",
                "clarification": "clarification",
                "routing_decision": "routing_decision",
                "end": END
            }
        )
        
        logger.debug("Edges configured")
        
        compiled_graph = graph.compile(
            name="supervisor_graph",
            checkpointer=memory
        )
        
        logger.info("Supervisor graph compiled successfully")
        return compiled_graph