
# src/modules/customer_service/integration.py
from typing import Dict, Any
from src.modules.customer_service.state import CustomerServiceAdapter, CustomerServiceState
from src.modules.customer_service.tools import CustomerServiceToolkit
from src.core.states.base_state import ChatbotState
from src.services.app_logger import log
from langchain_core.messages import AIMessage

class CustomerServiceModule:
    """Integration layer for Customer Service module with state management"""
    
    def __init__(self, llm):
        self.llm = llm
        self.adapter = CustomerServiceAdapter()
        self.toolkit = CustomerServiceToolkit()
        self.tools = self.toolkit.get_tools()
        self.logger = log.get(module="customer_service_integration")
    
    def process_request(self, chatbot_state: ChatbotState) -> ChatbotState:
        """Process Customer Service request with state transformation"""
        try:
            # Transform to module-specific state
            cs_state = self.adapter.transform_to_module_state(chatbot_state)
            
            # Validate state
            if not self.adapter.validate_state(cs_state):
                self.logger.error("Invalid Customer Service state")
                return self._create_error_response(chatbot_state, "Invalid request format")
            
            # Process based on intent
            processed_state = self._process_by_intent(cs_state)
            
            # Transform back to chatbot state
            final_state = self.adapter.transform_to_chatbot_state(processed_state)
            
            self.logger.info("Customer Service request processed successfully",
                           intent=cs_state.get("intent"),
                           escalation_needed=cs_state.get("escalation_needed"))
            return final_state
            
        except Exception as e:
            self.logger.error("Customer Service processing failed", error=str(e))
            return self._create_error_response(chatbot_state, f"Processing error: {str(e)}")
    
    def _process_by_intent(self, state: CustomerServiceState) -> CustomerServiceState:
        """Process request based on detected intent"""
        intent = state.get("intent", "general")
        
        if intent == "customer_service":
            return self._handle_general_support(state)
        elif intent == "order_status":
            return self._handle_order_inquiry(state)
        elif intent == "payment_support":
            return self._handle_payment_issue(state)
        elif intent == "product_inquiry":
            return self._handle_product_inquiry(state)
        else:
            return self._handle_general_inquiry(state)
    
    def _handle_general_support(self, state: CustomerServiceState) -> CustomerServiceState:
        """Handle general customer support requests"""
        resolution_steps = [
            "Sorununuz kaydedildi ve inceleme altına alındı",
            "Müşteri bilgileriniz kontrol ediliyor",
            "Uygun çözüm seçenekleri belirleniyor",
            "24 saat içinde geri dönüş yapılacak"
        ]
        
        # Create ticket if customer ID available
        if state.get("customer_info") and state["customer_info"].customer_id:
            ticket_tool = next(tool for tool in self.tools if tool.name == "create_support_ticket")
            ticket_result = ticket_tool.invoke({
                "customer_id": state["customer_info"].customer_id,
                "title": state.get("user_prompt", "General Support Request")[:50],
                "description": state.get("user_prompt", ""),
                "category": state.get("issue_category", "general_inquiry"),
                "priority": state["ticket_info"].priority.value if state.get("ticket_info") and state["ticket_info"].priority else "medium"
            })
            
            if ticket_result.get("success"):
                if state.get("ticket_info"):
                    state["ticket_info"].ticket_id = ticket_result["ticket_id"]
                resolution_steps.insert(0, f"Destek talebi oluşturuldu: {ticket_result['ticket_id']}")
        
        state["resolution_steps"] = resolution_steps
        return state
    
    def _handle_order_inquiry(self, state: CustomerServiceState) -> CustomerServiceState:
        """Handle order status inquiries"""
        resolution_steps = []
        
        if state.get("order_id"):
            # Look up order status
            order_tool = next(tool for tool in self.tools if tool.name == "lookup_order_status")
            order_result = order_tool.invoke({"order_id": state["order_id"]})
            
            if order_result.get("success"):
                order_data = order_result["order_data"]
                resolution_steps = [
                    f"Sipariş ID: {order_data['order_id']}",
                    f"Durum: {order_data['status']}",
                    f"Sipariş Tarihi: {order_data['order_date']}",
                    f"Toplam Tutar: {order_data['total_amount']}"
                ]
                
                if order_data.get("tracking_number"):
                    resolution_steps.append(f"Kargo Takip No: {order_data['tracking_number']}")
                
                if order_data.get("estimated_delivery"):
                    resolution_steps.append(f"Tahmini Teslimat: {order_data['estimated_delivery']}")
            else:
                resolution_steps = [
                    "Sipariş bilgisi bulunamadı",
                    "Lütfen sipariş numaranızı kontrol edin",
                    "Destek ekibi ile iletişime geçin"
                ]
        else:
            resolution_steps = [
                "Sipariş durumu sorgusu için sipariş numarası gerekli",
                "Lütfen sipariş numaranızı paylaşın"
            ]
        
        state["resolution_steps"] = resolution_steps
        return state
    
    def _handle_payment_issue(self, state: CustomerServiceState) -> CustomerServiceState:
        """Handle payment related issues"""
        resolution_steps = [
            "Ödeme sorunu tespit edildi",
            "Finansal işlemler departmanına yönlendirildi",
            "2 iş günü içinde detaylı inceleme yapılacak",
            "Sonuç e-posta ile bildirilecek"
        ]
        
        # Escalate payment issues
        state["escalation_needed"] = True
        
        state["resolution_steps"] = resolution_steps
        return state
    
    def _handle_product_inquiry(self, state: CustomerServiceState) -> CustomerServiceState:
        """Handle product related inquiries"""
        resolution_steps = [
            "Ürün bilgileri hazırlanıyor",
            "Stok durumu kontrol ediliyor",
            "Fiyat ve kampanya bilgileri güncel olarak paylaşılacak",
            "Ürün uzmanımız size detaylı bilgi verecek"
        ]
        
        state["resolution_steps"] = resolution_steps
        return state
    
    def _handle_general_inquiry(self, state: CustomerServiceState) -> CustomerServiceState:
        """Handle general inquiries"""
        resolution_steps = [
            "Talebiniz alındı",
            "Uygun departmana yönlendiriliyor",
            "En kısa sürede geri dönüş yapılacak"
        ]
        
        state["resolution_steps"] = resolution_steps
        return state
    
    def _create_error_response(self, original_state: ChatbotState, error_message: str) -> ChatbotState:
        """Create error response while preserving message history"""
        error_response = AIMessage(
            content=f"❌ **Müşteri Hizmetleri Hatası**\n\n{error_message}\n\nLütfen tekrar deneyin veya destek ekibimiz ile iletişime geçin."
        )
        
        return ChatbotState(
            messages=original_state["messages"] + [error_response]
        )
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information and capabilities"""
        return {
            "name": "Customer Service",
            "version": "1.0.0",
            "description": "Comprehensive customer service management with ticket handling and escalation",
            "capabilities": [
                "Customer information lookup",
                "Support ticket creation and management",
                "Order status inquiries",
                "Payment issue handling",
                "Automatic escalation based on sentiment and priority",
                "Email notifications"
            ],
            "supported_intents": [
                "customer_service",
                "order_status", 
                "payment_support",
                "product_inquiry",
                "general_inquiry"
            ],
            "state_fields": [
                "user_prompt",
                "customer_info",
                "ticket_info",
                "intent",
                "sentiment_score",
                "order_id",
                "product_id",
                "issue_category",
                "resolution_steps",
                "escalation_needed"
            ]
        }
