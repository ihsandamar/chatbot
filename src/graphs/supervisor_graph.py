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
# Text2SQL import will be done dynamically to avoid circular imports

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

class ChatModeConfig(BaseModel):
    """Configuration for daily chat mode"""
    enabled: bool = Field(default=False, description="Enable daily chat format")
    system_prompt: str = Field(default="", description="Custom system prompt for daily chat")
    casual_responses: bool = Field(default=True, description="Use casual conversational responses")
    context_aware: bool = Field(default=True, description="Remember conversation context")
    personalization: bool = Field(default=False, description="Enable personalized responses")

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
                    "select", "veri", "data", "kayıt", "record", "dynamic-reporting", 
                    "dinamik", "rapor", "reporting"
                ],
                "english_keywords": [
                    "sql", "query", "database", "table", "select", "data", "record",
                    "show me", "list", "find", "search", "dynamic-reporting", "dynamic",
                    "reporting"
                ],
                "context_patterns": [
                    r"select\s+.*\s+from",  # SQL patterns
                    r"tablo.*göster|show.*table",
                    r"listele|list\s+all",
                    r"dynamic[-_]reporting|dinamik.*rapor"  # Dynamic reporting patterns
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
    
    def __init__(self, llm: LLM, chat_mode_config: Optional[ChatModeConfig] = None):
        super().__init__(llm=llm, state_class=State)
        self.logger = log.get(module="supervisor_graph", cls="SupervisorGraph")
        self.intent_detector = IntentDetector()
        self.extractor = message_extractor
        self.transformer = state_transformer
        
        # Chat mode configuration
        if isinstance(chat_mode_config, dict):
            # Convert dict to ChatModeConfig object
            self.chat_mode_config = ChatModeConfig(**chat_mode_config)
        elif chat_mode_config is None:
            self.chat_mode_config = ChatModeConfig()
        else:
            self.chat_mode_config = chat_mode_config
        
        # Initialize routing model with structured output
        self._setup_routing_model()
        
        # Add daily chat node if enabled
        self._daily_chat_enabled = self.chat_mode_config.enabled
        
        # Initialize Text2SQL subgraph lazily to avoid circular imports
        self.text2sql_graph = None
        self._text2sql_initialized = False
        
        self.logger.info("SupervisorGraph initialized", 
                        daily_chat_enabled=self._daily_chat_enabled)
    
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
            
            # Add simplified nodes
            graph.add_node("welcome", self.welcome_node)
            graph.add_node("module_selection", self.module_selection_node)
            graph.add_node("module_confirmation", self.module_confirmation_node)
            graph.add_node("await_user_prompt", self.await_user_prompt_node)
            graph.add_node("process_module_request", self.process_module_request_node)
            graph.add_node("error_handler", self.error_handler_node)
            
            self.logger.debug("Nodes added to graph")
            
            # Define simplified linear flow
            graph.add_edge(START, "welcome")
            graph.add_edge("welcome", "module_selection")
            graph.add_edge("module_selection", "module_confirmation")
            graph.add_edge("module_confirmation", "await_user_prompt")
            graph.add_edge("await_user_prompt", "process_module_request")
            graph.add_edge("process_module_request", END)
            graph.add_edge("error_handler", END)
            
            self.logger.debug("Edges defined")
            
            compiled_graph = graph.compile(
                name="supervisor_graph",
                checkpointer=memory
            )
            
            self.logger.info("Supervisor graph compiled successfully", 
                           daily_chat_enabled=self._daily_chat_enabled)
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build supervisor graph", error=str(e))
            raise
    
    def welcome_node(self, state: State) -> State:
        """Welcome users and show module selection options"""
        try:
            self.logger.debug("Welcome node executed")
            
            messages = state["messages"]
            
            # Always show module selection menu
            welcome_message = AIMessage(content="""🎯 **ERP Yardım Sistemi'ne Hoş Geldiniz!**

Lütfen yapmak istediğiniz işlemi seçin:

**🔢 1. Raporlar** - Satış raporları, ciro analizleri, grafikler
**🛠️ 2. Teknik Destek** - Sistem sorunları, hata çözümü, yardım
**📊 3. Veritabanı Sorguları** - SQL sorguları, data analizi, dinamik raporlar
**👥 4. Müşteri Hizmetleri** - Sipariş takibi, ödeme desteği
**💡 5. Özellik Talepleri** - Yeni özellik önerileri, geliştirme
**📚 6. Dokümantasyon** - Kullanım kılavuzları, eğitim materyalleri
**🏢 7. Şirket Bilgileri** - İletişim, genel bilgiler

**Seçim yapmak için:**
• Numarayı yazın (örn: "1" veya "3")
• Anahtar kelimesini yazın (örn: "raporlar", "sql", "destek")

Hangi modülü seçmek istiyorsunuz?""")
            
            return {
                **state,
                "messages": messages + [welcome_message],
                "workflow_step": "awaiting_module_selection",
                "available_modules": [
                    {"id": "1", "name": "raporlar", "module": ModuleType.REPORTING},
                    {"id": "2", "name": "destek", "module": ModuleType.SUPPORT},
                    {"id": "3", "name": "sql", "module": ModuleType.TEXT2SQL},
                    {"id": "4", "name": "müşteri", "module": ModuleType.CUSTOMER_SERVICE},
                    {"id": "5", "name": "talep", "module": ModuleType.REQUEST},
                    {"id": "6", "name": "dokuman", "module": ModuleType.DOCUMENTS_TRAINING},
                    {"id": "7", "name": "şirket", "module": ModuleType.COMPANY_INFO}
                ]
            }
            
        except Exception as e:
            self.logger.error("Welcome node failed", error=str(e))
            return self.error_handler_node(state)
    
    def module_selection_node(self, state: State) -> State:
        """Parse user selection and identify chosen module"""
        try:
            self.logger.debug("Module selection node executed")
            
            messages = state["messages"]
            user_message = self.extractor.extract_user_prompt(messages)
            available_modules = state.get("available_modules", [])
            
            # Parse user input
            selected_module = None
            user_input = user_message.lower().strip()
            
            # Check for number selection
            if user_input.isdigit():
                module_id = user_input
                for module in available_modules:
                    if module["id"] == module_id:
                        selected_module = module
                        break
            
            # Check for keyword selection
            if not selected_module:
                for module in available_modules:
                    if module["name"] in user_input or module["name"].replace("ü", "u").replace("ş", "s") in user_input:
                        selected_module = module
                        break
                
                # Additional keyword matching
                keyword_mapping = {
                    "rapor": "raporlar",
                    "report": "raporlar", 
                    "analiz": "raporlar",
                    "support": "destek",
                    "yardım": "destek",
                    "help": "destek",
                    "database": "sql",
                    "veritaban": "sql",
                    "sorgu": "sql",
                    "query": "sql",
                    "customer": "müşteri",
                    "musteri": "müşteri",
                    "siparis": "müşteri",
                    "request": "talep",
                    "öneri": "talep",
                    "feature": "talep",
                    "doc": "dokuman",
                    "egitim": "dokuman",
                    "training": "dokuman",
                    "company": "şirket",
                    "sirket": "şirket",
                    "iletisim": "şirket"
                }
                
                for keyword, module_name in keyword_mapping.items():
                    if keyword in user_input:
                        for module in available_modules:
                            if module["name"] == module_name:
                                selected_module = module
                                break
                        if selected_module:
                            break
            
            if selected_module:
                return {
                    **state,
                    "messages": messages,
                    "workflow_step": "module_selected",
                    "selected_module": selected_module,
                    "selected_module_type": selected_module["module"]
                }
            else:
                # Invalid selection
                error_message = AIMessage(content="""❌ **Geçersiz seçim!**

Lütfen aşağıdakilerden birini seçin:
• **1-7 arası bir sayı** (örn: "3")
• **Anahtar kelime** (örn: "sql", "raporlar", "destek")

Hangi modülü seçmek istiyorsunuz?""")
                
                return {
                    **state,
                    "messages": messages + [error_message],
                    "workflow_step": "invalid_selection"
                }
                
        except Exception as e:
            self.logger.error("Module selection failed", error=str(e))
            return self.error_handler_node(state)
    
    def module_confirmation_node(self, state: State) -> State:
        """Confirm module selection and provide information"""
        try:
            self.logger.debug("Module confirmation node executed")
            
            messages = state["messages"]
            selected_module = state.get("selected_module", {})
            workflow_step = state.get("workflow_step", "")
            
            # Handle invalid selection
            if workflow_step == "invalid_selection":
                return {
                    **state,
                    "workflow_step": "awaiting_module_selection"
                }
            
            if not selected_module:
                return self.error_handler_node(state)
            
            module_type = selected_module["module"]
            
            # Create confirmation message with module-specific information
            if module_type == ModuleType.REPORTING:
                confirmation_content = """✅ **Raporlama Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:
• 📊 Satış raporları görüntüleme
• 💰 Ciro analizleri
• 📈 Performans grafikleri
• 📋 Özel raporlar oluşturma

Raporlama sistemi başlatılıyor..."""

            elif module_type == ModuleType.SUPPORT:
                confirmation_content = """✅ **Teknik Destek Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:  
• 🔧 Sistem sorunlarını çözme
• ❓ Kullanım yardımı alma
• 🐛 Hata raporlama
• 📞 Teknik destek talep etme

Teknik destek sistemi başlatılıyor..."""

            elif module_type == ModuleType.TEXT2SQL:
                confirmation_content = """✅ **Veritabanı Sorguları Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:
• 🗃️ Veritabanı sorgulama
• 📊 SQL sorguları oluşturma  
• 📈 Dinamik raporlar
• 🔍 Veri analizi

Veritabanı sorgu sistemi başlatılıyor..."""

            elif module_type == ModuleType.CUSTOMER_SERVICE:
                confirmation_content = """✅ **Müşteri Hizmetleri Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:
• 📦 Sipariş durumu sorgulama
• 💳 Ödeme desteği
• 👤 Müşteri bilgileri
• 📞 Müşteri desteği

Müşteri hizmetleri sistemi başlatılıyor..."""

            elif module_type == ModuleType.REQUEST:
                confirmation_content = """✅ **Özellik Talepleri Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:
• 💡 Yeni özellik önerme
• 🚀 Geliştirme talepleri
• 📝 İyileştirme önerileri
• 🔄 Geri bildirim verme

Özellik talepleri sistemi başlatılıyor..."""

            elif module_type == ModuleType.DOCUMENTS_TRAINING:
                confirmation_content = """✅ **Dokümantasyon Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:
• 📚 Kullanım kılavuzlarına erişim
• 🎓 Eğitim materyalleri
• 📖 Yardım dokümanları
• 🎥 Video eğitimler

Dokümantasyon sistemi başlatılıyor..."""

            elif module_type == ModuleType.COMPANY_INFO:
                confirmation_content = """✅ **Şirket Bilgileri Modülü Seçildi**

Bu modülde şunları yapabilirsiniz:
• 🏢 Şirket bilgileri
• 📧 İletişim bilgileri
• 🌐 Genel bilgiler
• 📍 Adres ve konum

Şirket bilgileri sistemi başlatılıyor..."""

            else:
                confirmation_content = """✅ **Modül Seçildi**

Seçiminiz işleme alınıyor..."""

            # Add prompt request to the confirmation
            confirmation_content += f"""

💬 **Şimdi ne yapmak istediğinizi belirtin:**
Seçtiğiniz {selected_module['name']} modülü ile ilgili sorularınızı yazabilirsiniz."""

            confirmation_message = AIMessage(content=confirmation_content)
            
            return {
                **state,
                "messages": messages + [confirmation_message],
                "workflow_step": "module_confirmed",
                "confirmed_module": selected_module["module"]
            }
            
        except Exception as e:
            self.logger.error("Module confirmation failed", error=str(e))
            return self.error_handler_node(state)
    
    def await_user_prompt_node(self, state: State) -> State:
        """Wait for user to provide their specific request for the selected module"""
        try:
            self.logger.debug("Await user prompt node executed")
            
            # This node just passes through - the user input will come in the next message
            # We just store that we're awaiting a prompt
            return {
                **state,
                "workflow_step": "awaiting_user_prompt"
            }
            
        except Exception as e:
            self.logger.error("Await user prompt failed", error=str(e))
            return self.error_handler_node(state)
    
    def process_module_request_node(self, state: State) -> State:
        """Process the user's request within the selected module"""
        try:
            self.logger.debug("Process module request node executed")
            
            messages = state["messages"]
            confirmed_module = state.get("confirmed_module")
            user_message = self.extractor.extract_user_prompt(messages)
            
            if not confirmed_module:
                return self.error_handler_node(state)
            
            # Process based on selected module
            if confirmed_module == ModuleType.TEXT2SQL:
                return self._process_text2sql_request(state, user_message)
            elif confirmed_module == ModuleType.REPORTING:
                return self._process_reporting_request(state, user_message)
            elif confirmed_module == ModuleType.SUPPORT:
                return self._process_support_request(state, user_message)
            elif confirmed_module == ModuleType.CUSTOMER_SERVICE:
                return self._process_customer_service_request(state, user_message)
            elif confirmed_module == ModuleType.REQUEST:
                return self._process_feature_request(state, user_message)
            elif confirmed_module == ModuleType.DOCUMENTS_TRAINING:
                return self._process_documentation_request(state, user_message)
            elif confirmed_module == ModuleType.COMPANY_INFO:
                return self._process_company_info_request(state, user_message)
            else:
                return self._process_general_request(state, user_message)
            
        except Exception as e:
            self.logger.error("Process module request failed", error=str(e))
            return self.error_handler_node(state)
    
    def _process_text2sql_request(self, state: State, user_message: str) -> State:
        """Process Text2SQL requests using the subgraph"""
        try:
            self.logger.info("Processing Text2SQL request", request=user_message)
            
            # Initialize Text2SQL graph if not done yet
            self._initialize_text2sql_graph()
            
            messages = state["messages"]
            
            # Convert State to TestState for Text2SQL graph
            from src.graphs.text2sql_graph import TestState
            
            text2sql_state = {
                "user_query": user_message,
                "all_tables": [],
                "relevant_tables": [],
                "table_schemas": "",
                "generated_sql": "",
                "validated_sql": "",
                "is_valid": False,
                "sql_result": "",
                "is_error": False,
                "error_message": "",
                "fixed_sql": "",
                "debug_info": {
                    "supervisor_state": state,
                    "original_user_query": user_message
                }
            }
            
            # Execute Text2SQL graph
            compiled_graph = self.text2sql_graph.build_graph()
            thread_config = {"configurable": {"thread_id": "text2sql_execution"}}
            result = compiled_graph.invoke(text2sql_state, config=thread_config)
            
            # Format result
            sql_result = result.get("sql_result", "No result")
            generated_sql = result.get("generated_sql", "")
            is_error = result.get("is_error", False)
            
            if is_error:
                error_message = result.get("error_message", "Unknown error")
                content = f"""❌ **SQL Sorgu Hatası**

**Hata:** {error_message}

**Oluşturulan SQL:** 
```sql
{generated_sql}
```

Sorgunuzu daha net bir şekilde ifade edebilir misiniz?"""
            else:
                content = f"""✅ **SQL Sorgu Sonucu**

**Oluşturulan SQL:** 
```sql
{generated_sql}
```

**Sonuç:** 
```
{sql_result}
```

SQL sorgunuz başarıyla çalıştırıldı!"""

            response_message = AIMessage(content=content)
            
            return {
                **state,
                "messages": messages + [response_message],
                "workflow_step": "request_processed",
                "sql_result": sql_result,
                "generated_sql": generated_sql
            }
            
        except Exception as e:
            self.logger.error("Text2SQL processing failed", error=str(e))
            error_message = AIMessage(content=f"""❌ **SQL İşleme Hatası**

Veritabanı sorgusu işlenirken bir hata oluştu: {str(e)}

Lütfen sorgunuzu tekrar deneyin.""")
            
            return {
                **state,
                "messages": state["messages"] + [error_message],
                "workflow_step": "request_error"
            }
    
    def _process_reporting_request(self, state: State, user_message: str) -> State:
        """Process reporting requests"""
        response_message = AIMessage(content=f"""📊 **Raporlama Modülü**

Talebiniz: "{user_message}"

Bu modül henüz geliştirme aşamasındadır. Yakında:
• Satış raporları
• Ciro analizleri  
• Performans grafikleri
• Özel raporlar

özelliklerini kullanabileceksiniz.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def _process_support_request(self, state: State, user_message: str) -> State:
        """Process technical support requests"""
        response_message = AIMessage(content=f"""🛠️ **Teknik Destek Modülü**

Talebiniz: "{user_message}"

Bu modül henüz geliştirme aşamasındadır. Yakında:
• Sistem sorunları çözümü
• Kullanım yardımı
• Hata raporlama
• Teknik destek

özelliklerini kullanabileceksiniz.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def _process_customer_service_request(self, state: State, user_message: str) -> State:
        """Process customer service requests"""
        response_message = AIMessage(content=f"""👥 **Müşteri Hizmetleri Modülü**

Talebiniz: "{user_message}"

Bu modül henüz geliştirme aşamasındadır. Yakında:
• Sipariş durumu sorgulama
• Ödeme desteği
• Müşteri bilgileri
• Müşteri desteği

özelliklerini kullanabileceksiniz.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def _process_feature_request(self, state: State, user_message: str) -> State:
        """Process feature requests"""
        response_message = AIMessage(content=f"""💡 **Özellik Talepleri Modülü**

Talebiniz: "{user_message}"

Bu modül henüz geliştirme aşamasındadır. Yakında:
• Yeni özellik önerme
• Geliştirme talepleri
• İyileştirme önerileri
• Geri bildirim

özelliklerini kullanabileceksiniz.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def _process_documentation_request(self, state: State, user_message: str) -> State:
        """Process documentation requests"""
        response_message = AIMessage(content=f"""📚 **Dokümantasyon Modülü**

Talebiniz: "{user_message}"

Bu modül henüz geliştirme aşamasındadır. Yakında:
• Kullanım kılavuzları
• Eğitim materyalleri
• Yardım dokümanları
• Video eğitimler

özelliklerini kullanabileceksiniz.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def _process_company_info_request(self, state: State, user_message: str) -> State:
        """Process company info requests"""
        response_message = AIMessage(content=f"""🏢 **Şirket Bilgileri Modülü**

Talebiniz: "{user_message}"

Bu modül henüz geliştirme aşamasındadır. Yakında:
• Şirket bilgileri
• İletişim bilgileri
• Genel bilgiler
• Adres ve konum

özelliklerini kullanabileceksiniz.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def _process_general_request(self, state: State, user_message: str) -> State:
        """Process general requests"""
        response_message = AIMessage(content=f"""🔄 **Genel Modül**

Talebiniz: "{user_message}"

Bu istek uygun bir modülde işlenecek.""")
        
        return {
            **state,
            "messages": state["messages"] + [response_message],
            "workflow_step": "request_processed"
        }
    
    def intent_detection_node(self, state: State) -> State:
        """Detect user intent using LLM and rule-based approaches"""
        try:
            self.logger.debug("Intent detection node executed")
            
            messages = state["messages"]
            user_message = self.extractor.extract_user_prompt(messages)
            
            # Update loop tracking
            loop_count = state.get("loop_count", 0) + 1
            visited_nodes = state.get("visited_nodes", []) + ["intent_detection"]
            
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
            
            # Store decision and tracking info in state
            return {
                **state,
                "routing_decision": rule_based_decision.dict(),
                "workflow_step": "intent_detected",
                "loop_count": loop_count,
                "visited_nodes": visited_nodes
            }
            
        except Exception as e:
            self.logger.error("Intent detection failed", error=str(e))
            return self.error_handler_node(state)
    
    def clarification_node(self, state: State) -> State:
        """Ask for clarification when intent is unclear"""
        try:
            self.logger.debug("Clarification node executed")
            
            messages = state["messages"]
            routing_decision = RoutingDecision(**state.get("routing_decision", {}))
            
            # Track clarification attempts
            clarification_count = state.get("clarification_count", 0) + 1
            visited_nodes = state.get("visited_nodes", []) + ["clarification"] 
            
            clarification_message = AIMessage(content=f"""🤔 **Anlayabilmek için biraz daha detay gerekiyor**

{routing_decision.suggested_response}

**Alternatif olarak şunları da yapabilirim:**
• 📊 Rapor ve analiz işlemleri
• 🔍 Veritabanı sorgulama
• 👥 Müşteri hizmetleri desteği
• 🎧 Teknik destek

Lütfen ne yapmak istediğinizi daha açık bir şekilde belirtin.""")
            
            return {
                **state,
                "messages": messages + [clarification_message],
                "workflow_step": "clarification_requested",
                "clarification_count": clarification_count,
                "visited_nodes": visited_nodes
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
            confirmed_module = state.get("confirmed_module")
            
            # Use confirmed_module from the new flow
            if not confirmed_module:
                # Fallback to old routing decision if available (backward compatibility)
                routing_decision_dict = state.get("routing_decision", {})
                if routing_decision_dict:
                    routing_decision = RoutingDecision(**routing_decision_dict)
                    confirmed_module = routing_decision.target_module
                else:
                    self.logger.warning("No target module specified for routing")
                    return self.error_handler_node(state)
            
            # Create module-specific routing message using confirmed module
            fake_routing_decision = RoutingDecision(
                target_module=confirmed_module,
                confidence=1.0,
                next_action=NextAction.ROUTE_TO_MODULE,
                suggested_response="Module selected by user"
            )
            routing_message = self._create_routing_message(fake_routing_decision)
            
            # Update state with module context
            updated_state = {
                **state,
                "messages": messages + [routing_message],
                "workflow_step": "routed_to_module",
                "target_module": confirmed_module.value,
                "routing_confidence": 1.0,
                "routing_timestamp": datetime.now().isoformat(),
                "redirect_to_text2sql": confirmed_module == ModuleType.TEXT2SQL
            }
            
            self.logger.info("Successfully routed to module", 
                           module=confirmed_module.value,
                           confidence=1.0)
            
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
    
    def daily_chat_node(self, state: State) -> State:
        """Daily chat format node for casual conversation"""
        try:
            self.logger.debug("Daily chat node executed")
            
            messages = state["messages"]
            user_message = self.extractor.extract_user_prompt(messages)
            
            # Use custom system prompt if provided
            system_prompt = self.chat_mode_config.system_prompt or """Sen günlük sohbet formatında konuşan, samimi ve yardımsever bir asistansın. 
            
Kullanıcılarla dostane bir dille konuş, emojiler kullan ve resmi ERP işlemlerini de günlük konuşma dilinde açıkla.
Teknik konuları basit bir şekilde anlat ve kullanıcıya rehberlik et."""
            
            # Generate casual response using LLM
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"Kullanıcı mesajı: {user_message}\n\nGünlük sohbet formatında, samimi bir şekilde yanıt ver.")
            ])
            
            chat_chain = chat_prompt | self.llm.get_chat()
            response = chat_chain.invoke({"user_message": user_message})
            
            chat_message = AIMessage(content=f"💬 **Günlük Sohbet Modu**\n\n{response.content}")
            
            return {
                "messages": messages + [chat_message],
                "workflow_step": "daily_chat_completed",
                "chat_mode_used": True
            }
            
        except Exception as e:
            self.logger.error("Daily chat node failed", error=str(e))
            return self.error_handler_node(state)
    
    def _determine_welcome_route(self, state: State) -> str:
        """Determine if we should go to daily chat or module selection"""
        try:
            messages = state["messages"]
            
            # Simple heuristic: if it's a casual greeting or general chat, use daily chat
            if len(messages) > 0:
                last_message = messages[-1].content.lower()
                casual_patterns = [
                    "merhaba", "selam", "nasılsın", "ne haber", "günaydın", "iyi akşamlar",
                    "hello", "hi", "how are you", "what's up", "good morning", "good evening"
                ]
                
                if any(pattern in last_message for pattern in casual_patterns):
                    self.logger.debug("Routing to daily chat for casual greeting")
                    return "daily_chat"
            
            # For business queries, go straight to module selection
            self.logger.debug("Routing to module selection for business query")
            return "module_selection"
            
        except Exception as e:
            self.logger.error("Failed to determine welcome route", error=str(e))
            return "module_selection"
    
    def determine_module_route(self, state: State) -> str:
        """Determine which module to route to based on confirmed selection"""
        try:
            confirmed_module = state.get("confirmed_module")
            workflow_step = state.get("workflow_step", "")
            
            # Handle invalid selection
            if workflow_step == "awaiting_module_selection":
                self.logger.debug("Routing back to show options for invalid selection")
                return "show_options"
            
            if not confirmed_module:
                self.logger.warning("No confirmed module found, routing to error handler")
                return "error_handler"
            
            # Route based on confirmed module
            if confirmed_module == ModuleType.TEXT2SQL:
                self.logger.debug("Routing to Text2SQL subgraph")
                return "text2sql_subgraph"
            else:
                self.logger.debug("Routing to module router", module=confirmed_module)
                return "module_router"
                
        except Exception as e:
            self.logger.error("Failed to determine module route", error=str(e))
            return "error_handler"
    
    def determine_next_node(self, state: State) -> str:
        """Determine the next node based on current state and routing decision"""
        try:
            workflow_step = state.get("workflow_step", "")
            routing_decision_dict = state.get("routing_decision", {})
            
            # Check for loop prevention
            loop_count = state.get("loop_count", 0)
            visited_nodes = state.get("visited_nodes", [])
            
            # Prevent infinite loops
            if loop_count > 5:
                self.logger.warning("Loop limit reached, ending conversation", loop_count=loop_count)
                return "end"
            
            # Prevent cycling through the same nodes repeatedly
            if len(visited_nodes) > 3:
                recent_nodes = visited_nodes[-3:]
                if len(set(recent_nodes)) == 1:  # Same node visited 3 times in a row
                    self.logger.warning("Detected cycling, ending conversation", recent_nodes=recent_nodes)
                    return "end"
            
            if not routing_decision_dict:
                self.logger.warning("No routing decision found, showing options")
                return "show_options"
            
            routing_decision = RoutingDecision(**routing_decision_dict)
            
            # Decision logic based on next_action
            if routing_decision.next_action == NextAction.ROUTE_TO_MODULE:
                # Check if it's a Text2SQL request
                if routing_decision.target_module == ModuleType.TEXT2SQL:
                    self.logger.debug("Routing to Text2SQL subgraph")
                    return "text2sql_subgraph"
                else:
                    self.logger.debug("Routing to module", module=routing_decision.target_module)
                    return "module_router"
            
            elif routing_decision.next_action == NextAction.CLARIFY_INTENT:
                # Prevent too many clarification attempts
                clarification_count = state.get("clarification_count", 0)
                if clarification_count >= 2:
                    self.logger.warning("Too many clarification attempts, showing options instead")
                    return "show_options"
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
• Dinamik raporlama desteği etkinleştiriliyor

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
    
    def _initialize_text2sql_graph(self):
        """Lazy initialization of Text2SQL graph"""
        if not self._text2sql_initialized:
            from src.graphs.text2sql_graph import Text2SQLGraph
            from src.services.config_loader import ConfigLoader
            from langchain_community.utilities import SQLDatabase
            
            config = ConfigLoader.load_config("config/text2sql_config.yaml")
            db = SQLDatabase.from_uri(config.database.uri)
            self.text2sql_graph = Text2SQLGraph(self.llm, db=db)
            self._text2sql_initialized = True
            self.logger.debug("Text2SQL graph initialized lazily")

    def text2sql_state_converter_node(self, state: State) -> Dict[str, Any]:
        """Convert supervisor State to Text2SQL TestState"""
        try:
            self.logger.debug("Converting state for Text2SQL subgraph")
            
            messages = state["messages"]
            user_message = self.extractor.extract_user_prompt(messages)
            
            # Convert State to TestState for Text2SQL graph
            from src.graphs.text2sql_graph import TestState
            
            # Create TestState with user query and debug info
            text2sql_state = {
                "user_query": user_message,
                "all_tables": [],
                "relevant_tables": [],
                "table_schemas": "",
                "generated_sql": "",
                "validated_sql": "",
                "is_valid": False,
                "sql_result": "",
                "is_error": False,
                "error_message": "",
                "fixed_sql": "",
                # Add debug tracking
                "debug_info": {
                    "supervisor_routing": state.get("routing_decision", {}),
                    "confidence_score": state.get("routing_confidence", 0.0),
                    "workflow_step": state.get("workflow_step", ""),
                    "original_user_query": user_message,
                    "session_id": state.get("session_id", "unknown"),
                    "supervisor_state": state  # Keep original state for response conversion
                }
            }
            
            self.logger.debug("State converted for Text2SQL subgraph")
            return text2sql_state
            
        except Exception as e:
            self.logger.error("Failed to convert state for Text2SQL", error=str(e))
            return {
                "user_query": "ERROR: State conversion failed",
                "is_error": True,
                "error_message": str(e)
            }

    def text2sql_response_node(self, text2sql_result: Dict[str, Any]) -> State:
        """Convert Text2SQL result back to supervisor State"""
        try:
            self.logger.debug("Converting Text2SQL result back to supervisor state")
            
            # Get original supervisor state from debug info
            original_state = text2sql_result.get("debug_info", {}).get("supervisor_state", {})
            messages = original_state.get("messages", [])
            
            # Format the result for display
            sql_result = text2sql_result.get("sql_result", "No result")
            generated_sql = text2sql_result.get("generated_sql", "")
            is_error = text2sql_result.get("is_error", False)
            
            if is_error:
                error_message = text2sql_result.get("error_message", "Unknown error")
                content = f"""❌ **SQL Sorgu Hatası**

**Hata:** {error_message}

**Oluşturulan SQL:** 
```sql
{generated_sql}
```

Lütfen sorgunuzu daha net bir şekilde belirtin."""
            else:
                content = f"""✅ **SQL Sorgu Sonucu**

**Oluşturulan SQL:** 
```sql
{generated_sql}
```

**Sonuç:** 
```
{sql_result}
```"""
            
            response_message = AIMessage(content=content)
            
            return {
                **original_state,
                "messages": messages + [response_message],
                "workflow_step": "text2sql_completed",
                "sql_result": sql_result,
                "generated_sql": generated_sql,
                "text2sql_debug": {
                    "execution_time": "calculated",
                    "tables_used": text2sql_result.get("relevant_tables", []),
                    "sql_validation": text2sql_result.get("is_valid", False),
                    "error_occurred": is_error
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to convert Text2SQL result", error=str(e))
            error_message = AIMessage(content=f"""❌ **Text2SQL Hata**

Veritabanı sorgusu işlenirken bir hata oluştu: {str(e)}

Lütfen sorgunuzu tekrar deneyin veya farklı bir şekilde ifade edin.""")
            
            original_state = text2sql_result.get("debug_info", {}).get("supervisor_state", {})
            messages = original_state.get("messages", [])
            
            return {
                **original_state,
                "messages": messages + [error_message],
                "workflow_step": "text2sql_error"
            }


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