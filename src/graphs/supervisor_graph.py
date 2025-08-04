# src/graphs/supervisor_graph.py
from typing import Literal, Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM, State
from src.core.messages.validators import message_extractor
from src.core.states.transformers import state_transformer
from src.services.app_logger import log
import re
from datetime import datetime

class ModuleType(Enum):
    """Available ERP modules for routing"""
    REPORTING = "reporting"          # Sales, revenue, analysis reports
    SUPPORT = "support"              # Technical support, help desk  
    DOCUMENTS_TRAINING = "documents" # Documentation and training materials
    REQUEST = "request"              # Feature requests, improvements
    COMPANY_INFO = "company"         # Company information
    TEXT2SQL = "text2sql"           # Database queries
    CUSTOMER_SERVICE = "customer_service"  # Customer support
    OTHER = "other"                  # General queries, external APIs

class NextAction(str, Enum):
    """Possible next actions for the supervisor"""
    ROUTE_TO_MODULE = "route_to_module"
    CLARIFY_INTENT = "clarify_intent"
    SHOW_OPTIONS = "show_options"
    WELCOME_USER = "welcome_user"
    END_CONVERSATION = "end_conversation"
    HANDLE_ERROR = "handle_error"

class RoutingDecision(BaseModel):
    """Structured output model for routing decisions"""
    next_action: NextAction = Field(description="The next action to take")
    target_module: Optional[ModuleType] = Field(default=None, description="Target module if routing")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the decision")
    reasoning: str = Field(description="Reasoning behind the decision")
    detected_keywords: List[str] = Field(default_factory=list, description="Keywords that influenced the decision")
    suggested_response: str = Field(description="Suggested response to the user")
    requires_clarification: bool = Field(default=False, description="Whether clarification is needed")

class IntentDetector:
    """Advanced intent detection with confidence scoring"""
    
    def __init__(self):
        self.logger = log.get(module="intent_detector")
        self.module_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[ModuleType, Dict[str, Any]]:
        """Initialize keyword patterns and weights for each module"""
        return {
            ModuleType.REPORTING: {
                "turkish_keywords": [
                    "rapor", "ciro", "satış", "analiz", "grafik", "özet", "istatistik",
                    "performans", "gelir", "kar", "zarar", "trend", "dashboard", "metrik"
                ],
                "english_keywords": [
                    "report", "revenue", "sales", "analysis", "chart", "summary", "statistics",
                    "performance", "income", "profit", "loss", "trend", "dashboard", "metric"
                ],
                "context_patterns": [
                    r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}",  # Date patterns
                    r"bugün|dün|bu\s+hafta|bu\s+ay|geçen\s+ay",  # Time references
                    r"şube|branch|departman|department"
                ],
                "weight": 1.0
            },
            ModuleType.SUPPORT: {
                "turkish_keywords": [
                    "yardım", "destek", "hata", "sorun", "nasıl", "problem", "çözüm",
                    "teknik", "arıza", "çalışmıyor", "bozuk", "fix", "onarım"
                ],
                "english_keywords": [
                    "help", "support", "error", "issue", "how to", "problem", "solution",
                    "technical", "bug", "broken", "not working", "fix", "repair"
                ],
                "context_patterns": [
                    r"error\s*:\s*\w+",  # Error messages
                    r"çalışmıyor|works?\s+not|doesn'?t\s+work"
                ],
                "weight": 0.9
            },
            ModuleType.TEXT2SQL: {
                "turkish_keywords": [
                    "sql", "sorgu", "query", "veritabanı", "database", "tablo", "table",
                    "select", "veri", "data", "kayıt", "record"
                ],
                "english_keywords": [
                    "sql", "query", "database", "table", "select", "data", "record",
                    "show me", "list", "find", "search"
                ],
                "context_patterns": [
                    r"select\s+.*\s+from",  # SQL patterns
                    r"tablo.*göster|show.*table",
                    r"listele|list\s+all"
                ],
                "weight": 1.0
            },
            ModuleType.CUSTOMER_SERVICE: {
                "turkish_keywords": [
                    "müşteri", "sipariş", "order", "teslimat", "delivery", "ödeme", "payment",
                    "fatura", "invoice", "iade", "return", "şikayet", "complaint"
                ],
                "english_keywords": [
                    "customer", "order", "delivery", "payment", "invoice", "return",
                    "complaint", "refund", "shipping", "billing"
                ],
                "context_patterns": [
                    r"sipariş\s+no|order\s+id|order\s+number",
                    r"müşteri\s+no|customer\s+id"
                ],
                "weight": 1.0
            },
            ModuleType.REQUEST: {
                "turkish_keywords": [
                    "özellik", "istek", "talep", "ekle", "geliştir", "yeni", "öneri",
                    "iyileştirme", "enhancement", "feature"
                ],
                "english_keywords": [
                    "feature", "request", "add", "improve", "enhance", "new", "suggestion",
                    "enhancement", "development"
                ],
                "context_patterns": [
                    r"eklenmeli|should\s+add",
                    r"geliştirilmeli|should\s+improve"
                ],
                "weight": 0.8
            },
            ModuleType.DOCUMENTS_TRAINING: {
                "turkish_keywords": [
                    "doküman", "dokümantasyon", "eğitim", "training", "kılavuz", "guide",
                    "manual", "tutorial", "nasıl", "öğren"
                ],
                "english_keywords": [
                    "document", "documentation", "training", "guide", "manual",
                    "tutorial", "how to", "learn"
                ],
                "context_patterns": [
                    r"nasıl\s+yapılır|how\s+to\s+do",
                    r"kılavuz|guide|manual"
                ],
                "weight": 0.7
            },
            ModuleType.COMPANY_INFO: {
                "turkish_keywords": [
                    "şirket", "company", "hakkında", "about", "bilgi", "info", "iletişim",
                    "contact", "adres", "address", "telefon", "phone"
                ],
                "english_keywords": [
                    "company", "about", "info", "information", "contact", "address",
                    "phone", "email", "location"
                ],
                "context_patterns": [
                    r"hakkında\s+bilgi|about\s+us",
                    r"şirket\s+bilgisi|company\s+info"
                ],
                "weight": 0.6
            }
        }
    
    def detect_intent(self, user_message: str, conversation_context: Dict[str, Any] = None) -> RoutingDecision:
        """Detect intent with confidence scoring"""
        try:
            user_message_lower = user_message.lower()
            scores = {}
            all_keywords = {}
            
            # Calculate scores for each module
            for module, config in self.module_patterns.items():
                score = 0
                found_keywords = []
                
                # Check Turkish keywords
                for keyword in config["turkish_keywords"]:
                    if keyword in user_message_lower:
                        score += config["weight"]
                        found_keywords.append(keyword)
                
                # Check English keywords
                for keyword in config["english_keywords"]:
                    if keyword in user_message_lower:
                        score += config["weight"]
                        found_keywords.append(keyword)
                
                # Check context patterns
                for pattern in config["context_patterns"]:
                    if re.search(pattern, user_message_lower):
                        score += config["weight"] * 1.5  # Context patterns get higher weight
                        found_keywords.append(f"pattern:{pattern}")
                
                scores[module] = score
                all_keywords[module] = found_keywords
            
            # Find best match
            best_module = max(scores, key=scores.get)
            best_score = scores[best_module]
            
            # Normalize confidence (simple approach)
            total_score = sum(scores.values())
            confidence = best_score / max(total_score, 1.0) if total_score > 0 else 0.0
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # Determine next action based on confidence
            if confidence >= 0.7:
                next_action = NextAction.ROUTE_TO_MODULE
                target_module = best_module
                reasoning = f"High confidence match for {best_module.value}"
            elif confidence >= 0.4:
                next_action = NextAction.CLARIFY_INTENT
                target_module = best_module
                reasoning = f"Medium confidence, suggesting {best_module.value} but asking for clarification"
            else:
                next_action = NextAction.SHOW_OPTIONS
                target_module = None
                reasoning = "Low confidence, showing all available options"
            
            # Generate suggested response
            suggested_response = self._generate_response(
                next_action, target_module, confidence, user_message
            )
            
            decision = RoutingDecision(
                next_action=next_action,
                target_module=target_module,
                confidence=confidence,
                reasoning=reasoning,
                detected_keywords=all_keywords.get(best_module, []),
                suggested_response=suggested_response,
                requires_clarification=(confidence < 0.7)
            )
            
            self.logger.debug("Intent detection completed", 
                            best_module=best_module.value,
                            confidence=confidence,
                            next_action=next_action.value)
            
            return decision
            
        except Exception as e:
            self.logger.error("Intent detection failed", error=str(e))
            return RoutingDecision(
                next_action=NextAction.HANDLE_ERROR,
                confidence=0.0,
                reasoning=f"Error in intent detection: {str(e)}",
                suggested_response="Üzgünüm, isteğinizi anlayamadım. Lütfen tekrar deneyin."
            )
    
    def _generate_response(self, action: NextAction, module: Optional[ModuleType], 
                         confidence: float, user_message: str) -> str:
        """Generate appropriate response based on routing decision"""
        if action == NextAction.ROUTE_TO_MODULE and module:
            return f"✅ **{module.value.title()} modülüne yönlendiriliyor...**\n\nİsteğiniz işleme alınıyor."
        
        elif action == NextAction.CLARIFY_INTENT and module:
            return f"""🤔 **Anladığım kadarıyla {module.value} ile ilgili bir talebiniz var.**

Doğru anladım mı? Eğer öyleyse devam edebiliriz, değilse lütfen daha detay verin."""
        
        elif action == NextAction.SHOW_OPTIONS:
            return """👋 **Size nasıl yardımcı olabilirim?**

Aşağıdaki konularda destek verebilirim:

📊 **Raporlama** - Ciro, satış ve analiz raporları
🎧 **Destek** - Teknik destek ve yardım
📚 **Dokümantasyon** - Kullanım kılavuzları ve eğitim
💡 **Özellik Talebi** - Yeni özellik önerileri
🏢 **Şirket Bilgileri** - İletişim ve genel bilgiler
🔍 **Veritabanı Sorguları** - SQL sorguları ve veri analizi
👥 **Müşteri Hizmetleri** - Sipariş ve müşteri desteği

Hangi konuda yardıma ihtiyacınız var?"""
        
        else:
            return "Bir sorun oluştu. Lütfen tekrar deneyin."

@register_graph("supervisor")
class SupervisorGraph(BaseGraph):
    """Supervisor graph that coordinates module routing and manages conversation flow"""
    
    def __init__(self, llm: LLM):
        super().__init__(llm=llm, state_class=State)
        self.logger = log.get(module="supervisor_graph", cls="SupervisorGraph")
        self.intent_detector = IntentDetector()
        self.extractor = message_extractor
        self.transformer = state_transformer
        
        # Initialize routing model with structured output
        self._setup_routing_model()
        
        self.logger.info("SupervisorGraph initialized")
    
    def _setup_routing_model(self):
        """Setup LLM-based routing model with tool binding"""
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen ERP sistemleri için akıllı bir yönlendirme asistanısın.

Kullanıcı mesajlarını analiz ederek en uygun modüle yönlendiriyorsun:

**Mevcut Modüller:**
- REPORTING: Raporlar, ciro, satış analizleri, grafikler
- SUPPORT: Teknik destek, hata çözümü, yardım
- TEXT2SQL: Veritabanı sorguları, SQL komutları
- CUSTOMER_SERVICE: Müşteri desteği, sipariş durumu, ödeme
- REQUEST: Özellik talepleri, geliştirme önerileri
- DOCUMENTS_TRAINING: Dokümantasyon, eğitim materyalleri
- COMPANY_INFO: Şirket bilgileri, iletişim
- OTHER: Genel sorular

**Yönergeler:**
1. Kullanıcı mesajını dikkatli analiz et
2. En uygun modülü belirle
3. Güven skorunu hesapla (0-1 arası)
4. Düşük güven durumunda açıklama iste
5. Her zaman yardımcı ve profesyonel ol

Karar verirken anahtar kelimeleri, bağlamı ve kullanıcının gerçek niyetini dikkate al."""),
            ("human", "Kullanıcı mesajı: {user_message}\n\nGeçmiş bağlam: {context}")
        ])
        
        self.routing_model = routing_prompt | self.llm.get_chat().bind_tools(
            [RoutingDecision], tool_choice="required"
        )
    
    def build_graph(self):
        """Build the supervisor graph with all nodes and routing logic"""
        try:
            self.logger.debug("Building supervisor graph")
            
            memory = MemorySaver()
            graph = StateGraph(State)
            
            # Add all nodes
            graph.add_node("welcome", self.welcome_node)
            graph.add_node("intent_detection", self.intent_detection_node)
            graph.add_node("clarification", self.clarification_node)
            graph.add_node("module_router", self.module_router_node)
            graph.add_node("show_options", self.show_options_node)
            graph.add_node("error_handler", self.error_handler_node)
            
            self.logger.debug("Nodes added to graph")
            
            # Define edges
            graph.add_edge(START, "welcome")
            graph.add_edge("welcome", "intent_detection")
            
            # Conditional routing from intent detection
            graph.add_conditional_edges(
                "intent_detection",
                self.determine_next_node,
                {
                    "clarification": "clarification",
                    "show_options": "show_options", 
                    "module_router": "module_router",
                    "error_handler": "error_handler",
                    "end": END
                }
            )
            
            # From clarification, go back to intent detection or route to module
            graph.add_conditional_edges(
                "clarification",
                self.determine_next_node,
                {
                    "intent_detection": "intent_detection",
                    "module_router": "module_router",
                    "show_options": "show_options",
                    "end": END
                }
            )
            
            # From show_options, wait for user choice then detect intent
            graph.add_edge("show_options", "intent_detection")
            
            # Module router and error handler end the conversation
            graph.add_edge("module_router", END)
            graph.add_edge("error_handler", END)
            
            self.logger.debug("Edges defined")
            
            compiled_graph = graph.compile(
                name="supervisor_graph",
                checkpointer=memory
            )
            
            self.logger.info("Supervisor graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build supervisor graph", error=str(e))
            raise
    
    def welcome_node(self, state: State) -> State:
        """Welcome new users with personalized greeting"""
        try:
            self.logger.debug("Welcome node executed")
            
            messages = state["messages"]
            
            # Check if this is a new conversation
            is_new_conversation = len(messages) <= 1
            
            if is_new_conversation:
                welcome_message = AIMessage(content=f"""👋 **Merhaba! ERP Asistanınıza Hoş Geldiniz**

Ben size aşağıdaki konularda yardımcı olabilir:

📊 **Raporlama & Analiz** - Satış, ciro ve performans raporları
🎧 **Teknik Destek** - Sistem sorunları ve çözümleri  
🔍 **Veritabanı Sorguları** - SQL sorguları ve veri analizi
👥 **Müşteri Hizmetleri** - Sipariş takibi ve müşteri desteği
💡 **Özellik Talepleri** - Sistem geliştirme önerileri
📚 **Dokümantasyon** - Kullanım kılavuzları ve eğitim
🏢 **Şirket Bilgileri** - İletişim ve genel bilgiler

**Ne yapmak istiyorsunuz?** Size nasıl yardımcı olabilirim?

*Örnek: "Bu ayın ciro raporunu göster" veya "Müşteri tablosunu listele"*""")
            else:
                # Returning user
                welcome_message = AIMessage(content="👋 **Tekrar hoş geldiniz!** Size nasıl yardımcı olabilirim?")
            
            return {
                "messages": messages + [welcome_message],
                "workflow_step": "welcomed"
            }
            
        except Exception as e:
            self.logger.error("Welcome node failed", error=str(e))
            return self.error_handler_node(state)
    
    def intent_detection_node(self, state: State) -> State:
        """Detect user intent using LLM and rule-based approaches"""
        try:
            self.logger.debug("Intent detection node executed")
            
            messages = state["messages"]
            user_message = self.extractor.extract_user_prompt(messages)
            
            # Get conversation context
            context = self.extractor.extract_conversation_context(messages)
            
            # Use rule-based intent detection
            rule_based_decision = self.intent_detector.detect_intent(user_message, context)
            
            # Also use LLM for comparison (optional enhancement)
            try:
                llm_response = self.routing_model.invoke({
                    "user_message": user_message,
                    "context": str(context)
                })
                
                # Extract routing decision from LLM response if available
                if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
                    llm_decision_raw = llm_response.tool_calls[0]["args"]
                    # Use LLM decision if confidence is higher
                    if llm_decision_raw.get("confidence", 0) > rule_based_decision.confidence:
                        self.logger.debug("Using LLM decision over rule-based")
                        # Create RoutingDecision from LLM response
                        rule_based_decision = RoutingDecision(**llm_decision_raw)
                
            except Exception as llm_error:
                self.logger.warning("LLM routing failed, using rule-based", error=str(llm_error))
            
            # Store decision in state
            state["routing_decision"] = rule_based_decision.dict()
            state["workflow_step"] = "intent_detected"
            
            self.logger.info("Intent detected", 
                           target_module=rule_based_decision.target_module,
                           confidence=rule_based_decision.confidence,
                           next_action=rule_based_decision.next_action)
            
            return state
            
        except Exception as e:
            self.logger.error("Intent detection failed", error=str(e))
            return self.error_handler_node(state)
    
    def clarification_node(self, state: State) -> State:
        """Ask for clarification when intent is unclear"""
        try:
            self.logger.debug("Clarification node executed")
            
            messages = state["messages"]
            routing_decision = RoutingDecision(**state.get("routing_decision", {}))
            
            clarification_message = AIMessage(content=f"""🤔 **Anlayabilmek için biraz daha detay gerekiyor**

{routing_decision.suggested_response}

**Alternatif olarak şunları da yapabilirim:**
• 📊 Rapor ve analiz işlemleri
• 🔍 Veritabanı sorgulama
• 👥 Müşteri hizmetleri desteği
• 🎧 Teknik destek

Lütfen ne yapmak istediğinizi daha açık bir şekilde belirtin.""")
            
            return {
                "messages": messages + [clarification_message],
                "workflow_step": "clarification_requested"
            }
            
        except Exception as e:
            self.logger.error("Clarification node failed", error=str(e))
            return self.error_handler_node(state)
    
    def show_options_node(self, state: State) -> State:
        """Show all available options to the user"""
        try:
            self.logger.debug("Show options node executed")
            
            messages = state["messages"]
            
            options_message = AIMessage(content="""🏢 **ERP Sistem Modülleri**

Hangi alanda yardıma ihtiyacınız var?

**📊 RAPORLAMA & ANALİZ**
• Ciro raporları ve satış analizleri
• Performans metrikleri ve grafikler
• *Örnek: "Bu ayın ciro raporunu göster"*

**🔍 VERİTABANI SORGULARI**
• SQL sorguları ve veri analizi
• Tablo listeleme ve veri arama
• *Örnek: "Müşteriler tablosunu listele"*

**👥 MÜŞTERİ HİZMETLERİ**
• Sipariş durumu sorgulama
• Müşteri destek talepleri
• *Örnek: "Sipariş durumumu öğrenmek istiyorum"*

**🎧 TEKNİK DESTEK**
• Sistem sorunları ve çözümleri
• Hata raporlama ve düzeltme
• *Örnek: "Sistem hatası alıyorum, yardım lazım"*

**💡 ÖZELLİK TALEBİ**
• Yeni özellik önerileri
• Sistem geliştirme fikirleri
• *Örnek: "Raporlara yeni filtre eklenebilir mi?"*

**📚 DOKÜMANTASYON**
• Kullanım kılavuzları
• Eğitim materyalleri
• *Örnek: "Rapor oluşturma nasıl yapılır?"*

**🏢 ŞİRKET BİLGİLERİ**
• İletişim bilgileri
• Genel şirket bilgileri
• *Örnek: "İletişim bilgilerinizi öğrenebilir miyim?"*

Yukarıdaki konulardan hangisinde yardıma ihtiyacınız var? Lütfen belirtin.""")
            
            return {
                "messages": messages + [options_message],
                "workflow_step": "options_shown"
            }
            
        except Exception as e:
            self.logger.error("Show options node failed", error=str(e))
            return self.error_handler_node(state)
    
    def module_router_node(self, state: State) -> State:
        """Route to the appropriate module"""
        try:
            self.logger.debug("Module router node executed")
            
            messages = state["messages"]
            routing_decision = RoutingDecision(**state.get("routing_decision", {}))
            
            if not routing_decision.target_module:
                self.logger.warning("No target module specified for routing")
                return self.error_handler_node(state)
            
            # Create module-specific routing message
            routing_message = self._create_routing_message(routing_decision)
            
            # Update state with module context
            updated_state = {
                "messages": messages + [routing_message],
                "workflow_step": "routed_to_module",
                "target_module": routing_decision.target_module.value,
                "routing_confidence": routing_decision.confidence,
                "routing_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("Successfully routed to module", 
                           module=routing_decision.target_module.value,
                           confidence=routing_decision.confidence)
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Module routing failed", error=str(e))
            return self.error_handler_node(state)
    
    def error_handler_node(self, state: State) -> State:
        """Handle errors gracefully with helpful messages"""
        try:
            self.logger.debug("Error handler node executed")
            
            messages = state["messages"]
            
            error_message = AIMessage(content="""❌ **Bir sorun oluştu**

Üzgünüm, isteğinizi işlerken bir hata oluştu. 

**Tekrar deneyebilirsiniz:**
• Sorunuzu daha açık bir şekilde ifade edin
• Aşağıdaki seçeneklerden birini kullanın:
  - "Raporlama yardımı"
  - "Veritabanı sorgusu"
  - "Müşteri desteği"
  - "Teknik destek"

**Veya direkt olarak:**
• "Yardım" yazarak tüm seçenekleri görün
• "Destek" yazarak teknik yardım alın

Size nasıl yardımcı olabilirim?""")
            
            return {
                "messages": messages + [error_message],
                "workflow_step": "error_handled",
                "error_handled": True
            }
            
        except Exception as e:
            self.logger.error("Error handler itself failed", error=str(e))
            # Fallback error message
            fallback_message = AIMessage(content="Sistem hatası oluştu. Lütfen tekrar deneyin.")
            return {
                "messages": state.get("messages", []) + [fallback_message],
                "workflow_step": "critical_error"
            }
    
    def determine_next_node(self, state: State) -> str:
        """Determine the next node based on current state and routing decision"""
        try:
            workflow_step = state.get("workflow_step", "")
            routing_decision_dict = state.get("routing_decision", {})
            
            if not routing_decision_dict:
                self.logger.warning("No routing decision found, showing options")
                return "show_options"
            
            routing_decision = RoutingDecision(**routing_decision_dict)
            
            # Decision logic based on next_action
            if routing_decision.next_action == NextAction.ROUTE_TO_MODULE:
                self.logger.debug("Routing to module", module=routing_decision.target_module)
                return "module_router"
            
            elif routing_decision.next_action == NextAction.CLARIFY_INTENT:
                self.logger.debug("Requesting clarification")
                return "clarification"
            
            elif routing_decision.next_action == NextAction.SHOW_OPTIONS:
                self.logger.debug("Showing options")
                return "show_options"
            
            elif routing_decision.next_action == NextAction.HANDLE_ERROR:
                self.logger.debug("Handling error")
                return "error_handler"
            
            elif routing_decision.next_action == NextAction.END_CONVERSATION:
                self.logger.debug("Ending conversation")
                return "end"
            
            else:
                self.logger.warning("Unknown next action, defaulting to show options")
                return "show_options"
                
        except Exception as e:
            self.logger.error("Failed to determine next node", error=str(e))
            return "error_handler"
    
    def _create_routing_message(self, routing_decision: RoutingDecision) -> AIMessage:
        """Create appropriate routing message based on target module"""
        module = routing_decision.target_module
        confidence = routing_decision.confidence
        
        if module == ModuleType.REPORTING:
            content = f"""📊 **Raporlama Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

Raporlama sistemi başlatılıyor...
• Ciro raporları hazırlanıyor
• Satış analizleri kontrol ediliyor
• Grafik ve istatistikler yükleniyor

Lütfen bekleyin..."""

        elif module == ModuleType.TEXT2SQL:
            content = f"""🔍 **Veritabanı Sorgu Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

SQL sorgu sistemi başlatılıyor...
• Veritabanı bağlantısı kontrol ediliyor
• Tablo şemaları yükleniyor
• Sorgu motoru hazırlanıyor

Sorgunuz işleme alınıyor..."""

        elif module == ModuleType.CUSTOMER_SERVICE:
            content = f"""👥 **Müşteri Hizmetleri Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

Müşteri destek sistemi başlatılıyor...
• Müşteri bilgileri kontrol ediliyor
• Sipariş durumları sorgulanıyor
• Destek kanalları hazırlanıyor

Talebiniz işleme alınıyor..."""

        elif module == ModuleType.SUPPORT:
            content = f"""🎧 **Teknik Destek Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

Teknik destek sistemi başlatılıyor...
• Sistem durumu kontrol ediliyor
• Hata logları inceleniyor
• Çözüm veritabanı hazırlanıyor

Sorununuz analiz ediliyor..."""

        elif module == ModuleType.REQUEST:
            content = f"""💡 **Özellik Talebi Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

Geliştirme talebi sistemi başlatılıyor...
• Mevcut özellikler kontrol ediliyor
• Geliştirme yol haritası gözden geçiriliyor
• Talep formu hazırlanıyor

Öneriniz değerlendiriliyor..."""

        elif module == ModuleType.DOCUMENTS_TRAINING:
            content = f"""📚 **Dokümantasyon Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

Dokümantasyon sistemi başlatılıyor...
• Kullanım kılavuzları aranıyor
• Eğitim materyalleri yükleniyor
• Video ve rehberler hazırlanıyor

Bilgilendirme materyalleri getiriliyor..."""

        elif module == ModuleType.COMPANY_INFO:
            content = f"""🏢 **Şirket Bilgileri Modülüne Yönlendiriliyor** (Güven: {confidence:.0%})

Kurumsal bilgi sistemi başlatılıyor...
• İletişim bilgileri yükleniyor
• Şirket profili hazırlanıyor
• Genel bilgiler derleniyor

Bilgiler getiriliyor..."""

        else:
            content = f"""🔄 **Genel Modüle Yönlendiriliyor** (Güven: {confidence:.0%})

Genel destek sistemi başlatılıyor...
Talebiniz uygun departmana yönlendiriliyor..."""

        return AIMessage(content=content)

# Unit Tests for Supervisor Graph
class TestSupervisorGraph:
    """Test cases for supervisor graph functionality"""
    
    def __init__(self):
        from src.models.models import LLM
        self.llm = LLM(model="gpt-4o-mini", temperature=0.0)
        self.supervisor = SupervisorGraph(self.llm)
        self.intent_detector = IntentDetector()
    
    def test_intent_detection_accuracy(self):
        """Test intent detection with various inputs"""
        test_cases = [
            {
                "input": "Bu ayın ciro raporunu göster",
                "expected_module": ModuleType.REPORTING,
                "min_confidence": 0.7
            },
            {
                "input": "Müşteriler tablosunu listele",
                "expected_module": ModuleType.TEXT2SQL,
                "min_confidence": 0.7
            },
            {
                "input": "Sipariş durumumu öğrenmek istiyorum", 
                "expected_module": ModuleType.CUSTOMER_SERVICE,
                "min_confidence": 0.7
            },
            {
                "input": "Sistem hatası alıyorum",
                "expected_module": ModuleType.SUPPORT,
                "min_confidence": 0.6
            },
            {
                "input": "Yeni özellik eklenebilir mi?",
                "expected_module": ModuleType.REQUEST,
                "min_confidence": 0.5
            }
        ]
        
        results = []
        for case in test_cases:
            decision = self.intent_detector.detect_intent(case["input"])
            passed = (
                decision.target_module == case["expected_module"] and
                decision.confidence >= case["min_confidence"]
            )
            results.append({
                "input": case["input"],
                "expected": case["expected_module"],
                "actual": decision.target_module,
                "confidence": decision.confidence,
                "passed": passed
            })
        
        return results
    
    def test_routing_logic_correctness(self):
        """Test routing logic with various scenarios"""
        from langchain_core.messages import HumanMessage, AIMessage
        
        test_state = State(
            messages=[
                HumanMessage(content="Bu ayın ciro raporunu göster"),
                AIMessage(content="Merhaba! Size nasıl yardımcı olabilirim?")
            ]
        )
        
        # Test intent detection
        result_state = self.supervisor.intent_detection_node(test_state)
        routing_decision = RoutingDecision(**result_state.get("routing_decision", {}))
        
        # Test next node determination
        next_node = self.supervisor.determine_next_node(result_state)
        
        return {
            "routing_decision": routing_decision.dict(),
            "next_node": next_node,
            "test_passed": (
                routing_decision.target_module == ModuleType.REPORTING and
                next_node == "module_router"
            )
        }
    
    def test_error_handling_robustness(self):
        """Test error handling scenarios"""
        # Test with malformed state
        malformed_state = {"invalid": "state"}
        
        try:
            result = self.supervisor.error_handler_node(malformed_state)
            error_handled = result.get("error_handled", False)
            has_error_message = len(result.get("messages", [])) > 0
            
            return {
                "error_handled": error_handled,
                "has_error_message": has_error_message,
                "test_passed": error_handled and has_error_message
            }
        except Exception as e:
            return {
                "error_handled": False,
                "exception": str(e),
                "test_passed": False
            }
    
    def run_all_tests(self):
        """Run all test cases"""
        print("🧪 Running Supervisor Graph Tests...")
        print("=" * 50)
        
        # Test 1: Intent Detection
        print("📋 Test 1: Intent Detection Accuracy")
        intent_results = self.test_intent_detection_accuracy()
        passed_intent = sum(1 for r in intent_results if r["passed"])
        print(f"✅ Passed: {passed_intent}/{len(intent_results)} test cases")
        for result in intent_results:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['input'][:30]}... -> {result['actual']} ({result['confidence']:.2f})")
        print()
        
        # Test 2: Routing Logic
        print("🔄 Test 2: Routing Logic Correctness")
        routing_result = self.test_routing_logic_correctness()
        status = "✅" if routing_result["test_passed"] else "❌"
        print(f"{status} Routing test passed: {routing_result['test_passed']}")
        print(f"  Next node: {routing_result['next_node']}")
        print()
        
        # Test 3: Error Handling
        print("⚠️ Test 3: Error Handling Robustness")
        error_result = self.test_error_handling_robustness()
        status = "✅" if error_result["test_passed"] else "❌"
        print(f"{status} Error handling test passed: {error_result['test_passed']}")
        print()
        
        print("🏁 All tests completed!")
        return {
            "intent_detection": intent_results,
            "routing_logic": routing_result,
            "error_handling": error_result
        }

# Example usage
if __name__ == "__main__":
    # Initialize and test
    test_suite = TestSupervisorGraph()
    test_results = test_suite.run_all_tests()
    
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print(f"Intent Detection: {sum(1 for r in test_results['intent_detection'] if r['passed'])}/{len(test_results['intent_detection'])} passed")
    print(f"Routing Logic: {'✅' if test_results['routing_logic']['test_passed'] else '❌'}")
    print(f"Error Handling: {'✅' if test_results['error_handling']['test_passed'] else '❌'}")