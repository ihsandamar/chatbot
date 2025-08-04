# src/core/chatbot_controller.py
from typing import Dict, Any, Optional, List
from src.core.states.base_state import ChatbotState
from src.core.states.state_registry import state_registry
from src.core.messages.validators import message_extractor, message_validator, ValidationLevel
from src.modules.text2sql.integration import Text2SQLModule
from src.modules.customer_service.integration import CustomerServiceModule
from src.services.app_logger import log
from langchain_core.messages import AIMessage, HumanMessage
from enum import Enum

class ModuleType(Enum):
    TEXT2SQL = "text2sql"
    CUSTOMER_SERVICE = "customer_service"
    GENERAL = "general"

class ERPChatbotController:
    """Main controller for ERP chatbot with unified state management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = log.get(module="erp_chatbot_controller")
        self.modules = self._initialize_modules()
        self.message_validator = message_validator
        self.message_extractor = message_extractor
        
        # Initialize state registry
        self._register_module_states()
        
        self.logger.info("ERP Chatbot Controller initialized", 
                        modules=list(self.modules.keys()),
                        registered_states=list(state_registry.list_states()))
    
    def _initialize_modules(self) -> Dict[str, Any]:
        """Initialize all available modules"""
        modules = {}
        
        try:
            # Initialize Text2SQL module if database config available
            if self.config.database and self.config.database.uri:
                from src.models.models import LLM
                llm = LLM(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    api_key=self.config.llm.api_key
                )
                modules[ModuleType.TEXT2SQL.value] = Text2SQLModule(
                    llm=llm,
                    db_uri=self.config.database.uri
                )
                self.logger.info("Text2SQL module initialized")
            
            # Initialize Customer Service module
            if self.config.llm:
                from src.models.models import LLM
                llm = LLM(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    api_key=self.config.llm.api_key
                )
                modules[ModuleType.CUSTOMER_SERVICE.value] = CustomerServiceModule(llm=llm)
                self.logger.info("Customer Service module initialized")
                
        except Exception as e:
            self.logger.error("Failed to initialize modules", error=str(e))
        
        return modules
    
    def _register_module_states(self):
        """Register all module states with the registry"""
        try:
            # States are automatically registered via decorators in module files
            # This method can be used for additional validation or manual registration
            for state_name in state_registry.list_states():
                if not state_registry.validate_compatibility(state_name):
                    self.logger.warning("State compatibility issue", state_name=state_name)
        except Exception as e:
            self.logger.error("Failed to register module states", error=str(e))
    
    def process_message(self, chatbot_state: ChatbotState) -> ChatbotState:
        """Main entry point for processing messages with unified state management"""
        try:
            # Validate input state
            validation_result = self.message_validator.validate_message_history(
                chatbot_state["messages"]
            )
            
            if not validation_result.is_valid:
                self.logger.error("Message validation failed", errors=validation_result.errors)
                return self._create_validation_error_response(chatbot_state, validation_result)
            
            # Determine appropriate module
            module_type = self._determine_module(chatbot_state)
            self.logger.debug("Module determined", module_type=module_type.value)
            
            # Process with appropriate module
            if module_type == ModuleType.TEXT2SQL and ModuleType.TEXT2SQL.value in self.modules:
                return self.modules[ModuleType.TEXT2SQL.value].process_request(chatbot_state)
            elif module_type == ModuleType.CUSTOMER_SERVICE and ModuleType.CUSTOMER_SERVICE.value in self.modules:
                return self.modules[ModuleType.CUSTOMER_SERVICE.value].process_request(chatbot_state)
            else:
                return self._handle_general_request(chatbot_state)
                
        except Exception as e:
            self.logger.error("Message processing failed", error=str(e))
            return self._create_error_response(chatbot_state, f"Processing error: {str(e)}")
    
    def _determine_module(self, chatbot_state: ChatbotState) -> ModuleType:
        """Determine which module should handle the request"""
        try:
            # Extract intent from messages
            intent = self.message_extractor.detect_intent(chatbot_state["messages"])
            self.logger.debug("Intent detected", intent=intent)
            
            # Map intents to modules
            intent_module_mapping = {
                "sql_query": ModuleType.TEXT2SQL,
                "customer_service": ModuleType.CUSTOMER_SERVICE,
                "order_status": ModuleType.CUSTOMER_SERVICE,
                "payment_support": ModuleType.CUSTOMER_SERVICE,
                "product_inquiry": ModuleType.CUSTOMER_SERVICE,
                "general": ModuleType.GENERAL
            }
            
            return intent_module_mapping.get(intent, ModuleType.GENERAL)
            
        except Exception as e:
            self.logger.error("Failed to determine module", error=str(e))
            return ModuleType.GENERAL
    
    def _handle_general_request(self, chatbot_state: ChatbotState) -> ChatbotState:
        """Handle general requests that don't fit specific modules"""
        user_prompt = self.message_extractor.extract_user_prompt(chatbot_state["messages"])
        
        response_content = f"""👋 **Merhaba!**

Size nasıl yardımcı olabilirim? Aşağıdaki konularda destek verebilirim:

🔍 **Veritabanı Sorguları:**
- "Müşteriler tablosunu göster"
- "Bu ayın satış raporunu çıkar"
- "Ürün stok durumunu sorgula"

🎧 **Müşteri Hizmetleri:**
- Sipariş durumu sorgulama
- Ödeme sorunları
- Ürün bilgileri
- Genel destek talepleri

📊 **Raporlama:**
- Ciro raporları
- Satış analizleri
- İşletme raporları

Lütfen ne yapmak istediğinizi belirtin.

**Sorununuz:** {user_prompt}"""
        
        response_message = AIMessage(content=response_content)
        
        return ChatbotState(
            messages=chatbot_state["messages"] + [response_message]
        )
    
    def _create_validation_error_response(self, chatbot_state: ChatbotState, validation_result) -> ChatbotState:
        """Create response for validation errors"""
        error_details = "\n".join([f"- {error}" for error in validation_result.errors])
        suggestions = self.message_validator.suggest_fixes(validation_result)
        suggestions_text = "\n".join([f"- {suggestion}" for suggestion in suggestions])
        
        response_content = f"""❌ **Mesaj Doğrulama Hatası**

**Tespit Edilen Hatalar:**
{error_details}

**Öneriler:**
{suggestions_text}

Lütfen mesajınızı düzelterek tekrar deneyin."""
        
        response_message = AIMessage(content=response_content)
        
        return ChatbotState(
            messages=chatbot_state["messages"] + [response_message]
        )
    
    def _create_error_response(self, chatbot_state: ChatbotState, error_message: str) -> ChatbotState:
        """Create generic error response"""
        response_content = f"""❌ **Sistem Hatası**

{error_message}

Lütfen tekrar deneyin veya sistem yöneticisi ile iletişime geçin."""
        
        response_message = AIMessage(content=response_content)
        
        return ChatbotState(
            messages=chatbot_state["messages"] + [response_message]
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "controller_status": "active",
            "modules": {},
            "state_registry": {
                "registered_states": list(state_registry.list_states()),
                "total_states": len(state_registry.list_states())
            },
            "configuration": {
                "has_database": bool(self.config.database),
                "has_llm": bool(self.config.llm),
                "validation_level": self.message_validator.validation_level.value
            }
        }
        
        # Get module status
        for module_name, module_instance in self.modules.items():
            try:
                status["modules"][module_name] = {
                    "status": "active",
                    "info": module_instance.get_module_info()
                }
            except Exception as e:
                status["modules"][module_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": self.message_extractor.extract_comprehensive_data([]).metadata.get("extraction_timestamp")
        }
        
        # Check message validator
        try:
            test_messages = [HumanMessage(content="test")]
            validation_result = self.message_validator.validate_message_history(test_messages)
            health["components"]["message_validator"] = {
                "status": "healthy" if validation_result.is_valid else "warning",
                "details": f"Validation passed: {validation_result.is_valid}"
            }
        except Exception as e:
            health["components"]["message_validator"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check state registry
        try:
            states_count = len(state_registry.list_states())
            health["components"]["state_registry"] = {
                "status": "healthy",
                "details": f"Registered states: {states_count}"
            }
        except Exception as e:
            health["components"]["state_registry"] = {
                "status": "error", 
                "error": str(e)
            }
        
        # Check modules
        for module_name, module_instance in self.modules.items():
            try:
                module_info = module_instance.get_module_info()
                health["components"][f"module_{module_name}"] = {
                    "status": "healthy",
                    "details": f"Version: {module_info.get('version', 'unknown')}"
                }
            except Exception as e:
                health["components"][f"module_{module_name}"] = {
                    "status": "error",
                    "error": str(e)
                }
                health["overall_status"] = "degraded"
        
        return health
