# src/modules/customer_service/state.py
from typing import Optional, Dict, Any, List
from src.core.states.base_state import ChatbotState
from src.core.states.state_registry import register_state
from src.core.states.state_adapter import StateAdapter
from src.core.messages.validators import message_extractor
from src.services.app_logger import log
from langchain_core.messages import AIMessage, HumanMessage
from enum import Enum
from dataclasses import dataclass

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketStatus(Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class CustomerInfo:
    customer_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    tier: Optional[str] = None  # VIP, Premium, Standard

@dataclass
class TicketInfo:
    ticket_id: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[TicketPriority] = None
    status: Optional[TicketStatus] = None

class CustomerServiceState(ChatbotState):
    """Extended state for Customer Service operations"""
    user_prompt: str
    customer_info: Optional[CustomerInfo] = None
    ticket_info: Optional[TicketInfo] = None
    intent: Optional[str] = None
    sentiment_score: Optional[float] = None
    order_id: Optional[str] = None
    product_id: Optional[str] = None
    issue_category: Optional[str] = None
    resolution_steps: Optional[List[str]] = None
    escalation_needed: Optional[bool] = None
    context_metadata: Optional[Dict[str, Any]] = None

class CustomerServiceAdapter(StateAdapter[CustomerServiceState]):
    """State adapter for Customer Service module"""
    
    def __init__(self, module_name: str = "customer_service"):
        super().__init__(module_name)
        self.extractor = message_extractor
    
    def get_state_class(self) -> type:
        """Return the CustomerServiceState class"""
        return CustomerServiceState
    
    def transform_to_module_state(self, chatbot_state: ChatbotState) -> CustomerServiceState:
        """Transform ChatbotState to CustomerServiceState"""
        try:
            messages = chatbot_state["messages"]
            
            # Extract comprehensive data
            extracted_data = self.extractor.extract_comprehensive_data(messages)
            
            # Build customer info
            customer_info = CustomerInfo(
                customer_id=extracted_data.customer_id,
                email=self._extract_customer_email(messages),
                phone=self._extract_customer_phone(messages),
                name=self._extract_customer_name(messages)
            )
            
            # Determine issue category and priority
            issue_category = self._categorize_issue(extracted_data.user_prompt)
            priority = self._determine_priority(extracted_data.user_prompt, extracted_data.intent)
            
            # Build ticket info
            ticket_info = TicketInfo(
                title=self._generate_ticket_title(extracted_data.user_prompt),
                category=issue_category,
                priority=priority,
                status=TicketStatus.NEW
            )
            
            # Calculate sentiment
            sentiment_score = self._analyze_sentiment(extracted_data.user_prompt)
            
            # Create enhanced state
            module_state = CustomerServiceState(
                messages=messages,
                user_prompt=extracted_data.user_prompt,
                customer_info=customer_info,
                ticket_info=ticket_info,
                intent=extracted_data.intent,
                sentiment_score=sentiment_score,
                order_id=extracted_data.order_id,
                product_id=extracted_data.product_id,
                issue_category=issue_category,
                escalation_needed=self._needs_escalation(sentiment_score, priority),
                context_metadata={
                    "entities": extracted_data.entities,
                    "conversation_context": extracted_data.metadata,
                    "extraction_timestamp": extracted_data.metadata.get("extraction_timestamp")
                }
            )
            
            self.logger.debug("Transformed to Customer Service state",
                            intent=extracted_data.intent,
                            issue_category=issue_category,
                            priority=priority.value if priority else None,
                            sentiment=sentiment_score)
            
            return module_state
            
        except Exception as e:
            self.logger.error("Failed to transform to Customer Service state", error=str(e))
            return CustomerServiceState(
                messages=chatbot_state["messages"],
                user_prompt="Error extracting user prompt"
            )
    
    def transform_to_chatbot_state(self, module_state: CustomerServiceState) -> ChatbotState:
        """Transform CustomerServiceState back to ChatbotState"""
        try:
            chatbot_state = ChatbotState(messages=module_state["messages"])
            
            # Add resolution response if available
            if module_state.get("resolution_steps") and not self._resolution_in_messages(module_state):
                response_content = self._format_customer_service_response(module_state)
                response_message = AIMessage(content=response_content)
                chatbot_state["messages"] = chatbot_state["messages"] + [response_message]
                
                self.logger.debug("Added customer service response to chatbot state")
            
            return chatbot_state
            
        except Exception as e:
            self.logger.error("Failed to transform to chatbot state", error=str(e))
            return ChatbotState(messages=module_state.get("messages", []))
    
    def validate_state(self, state: CustomerServiceState) -> bool:
        """Validate CustomerServiceState"""
        try:
            # Check required fields
            if not isinstance(state.get("messages"), list):
                return False
            
            if not isinstance(state.get("user_prompt"), str):
                return False
            
            # Validate sentiment score range
            if state.get("sentiment_score") is not None:
                sentiment = state["sentiment_score"]
                if not isinstance(sentiment, (int, float)) or not -1 <= sentiment <= 1:
                    self.logger.warning("Invalid sentiment score range")
                    return False
            
            # Validate enums
            if state.get("ticket_info"):
                ticket = state["ticket_info"]
                if hasattr(ticket, 'priority') and ticket.priority:
                    if not isinstance(ticket.priority, TicketPriority):
                        return False
                if hasattr(ticket, 'status') and ticket.status:
                    if not isinstance(ticket.status, TicketStatus):
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error("State validation failed", error=str(e))
            return False
    
    def _extract_customer_email(self, messages: List) -> Optional[str]:
        """Extract customer email from messages"""
        entities = self.extractor.extract_all_entities(messages)
        emails = entities.get("email", [])
        return emails[0] if emails else None
    
    def _extract_customer_phone(self, messages: List) -> Optional[str]:
        """Extract customer phone from messages"""
        entities = self.extractor.extract_all_entities(messages)
        phones = entities.get("phone", [])
        return phones[0] if phones else None
    
    def _extract_customer_name(self, messages: List) -> Optional[str]:
        """Extract customer name from messages using patterns"""
        # Simple name extraction patterns
        import re
        
        user_prompt = self.extractor.extract_user_prompt(messages)
        name_patterns = [
            r'(?:adım|ismim|ben)\s+([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)',
            r'name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, user_prompt, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _categorize_issue(self, user_prompt: str) -> str:
        """Categorize the customer issue"""
        user_prompt_lower = user_prompt.lower()
        
        categories = {
            "order_issue": ["sipariş", "order", "teslimat", "delivery", "kargo", "shipping"],
            "payment_issue": ["ödeme", "payment", "para", "money", "kredi", "credit", "fatura", "invoice"],
            "product_issue": ["ürün", "product", "kalite", "quality", "bozuk", "broken", "defective"],
            "account_issue": ["hesap", "account", "login", "şifre", "password", "profil", "profile"],
            "technical_issue": ["teknik", "technical", "sistem", "system", "hata", "error", "bug"],
            "general_inquiry": ["bilgi", "info", "soru", "question", "nasıl", "how"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in user_prompt_lower for keyword in keywords):
                return category
        
        return "general_inquiry"
    
    def _determine_priority(self, user_prompt: str, intent: str) -> TicketPriority:
        """Determine ticket priority based on content and intent"""
        user_prompt_lower = user_prompt.lower()
        
        # Urgent indicators
        urgent_keywords = ["acil", "urgent", "hemen", "immediately", "kritik", "critical"]
        if any(keyword in user_prompt_lower for keyword in urgent_keywords):
            return TicketPriority.URGENT
        
        # High priority indicators
        high_keywords = ["önemli", "important", "problem", "hata", "error", "çalışmıyor", "not working"]
        if any(keyword in user_prompt_lower for keyword in high_keywords):
            return TicketPriority.HIGH
        
        # Payment/order issues are typically medium priority
        if intent in ["payment_support", "order_status"]:
            return TicketPriority.MEDIUM
        
        return TicketPriority.LOW
    
    def _generate_ticket_title(self, user_prompt: str) -> str:
        """Generate a concise ticket title"""
        # Take first 50 characters and clean up
        title = user_prompt[:50].strip()
        if len(user_prompt) > 50:
            title += "..."
        return title
    
    def _analyze_sentiment(self, user_prompt: str) -> float:
        """Simple rule-based sentiment analysis"""
        user_prompt_lower = user_prompt.lower()
        
        # Negative sentiment indicators
        negative_words = [
            "kötü", "bad", "berbat", "terrible", "awful", "problem", "hata", "error",
            "sinirli", "angry", "memnun değil", "unsatisfied", "şikayet", "complaint"
        ]
        
        # Positive sentiment indicators  
        positive_words = [
            "iyi", "good", "mükemmel", "excellent", "teşekkür", "thank", "harika", "great",
            "memnun", "satisfied", "güzel", "nice", "beğendim", "like"
        ]
        
        positive_count = sum(1 for word in positive_words if word in user_prompt_lower)
        negative_count = sum(1 for word in negative_words if word in user_prompt_lower)
        
        # Simple scoring (-1 to 1)
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total_words = len(user_prompt_lower.split())
        sentiment_score = (positive_count - negative_count) / max(total_words * 0.1, 1)
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment_score))
    
    def _needs_escalation(self, sentiment_score: Optional[float], priority: Optional[TicketPriority]) -> bool:
        """Determine if ticket needs escalation"""
        if priority == TicketPriority.URGENT:
            return True
        
        if sentiment_score is not None and sentiment_score < -0.5:
            return True
        
        return False
    
    def _resolution_in_messages(self, state: CustomerServiceState) -> bool:
        """Check if resolution is already in message history"""
        if not state.get("resolution_steps"):
            return True
        
        messages = state["messages"]
        for msg in reversed(messages):
            if hasattr(msg, 'content') and "Resolution Steps" in str(msg.content):
                return True
        
        return False
    
    def _format_customer_service_response(self, state: CustomerServiceState) -> str:
        """Format customer service response"""
        response_parts = []
        
        # Greeting based on sentiment
        if state.get("sentiment_score", 0) < -0.3:
            response_parts.append("🙏 **Üzgünüz, sorununuzu anlıyoruz**")
        else:
            response_parts.append("👋 **Merhaba! Size yardımcı olmaktan mutluluk duyarız**")
        
        response_parts.append("")
        
        # Customer and ticket info
        if state.get("customer_info") and state["customer_info"].customer_id:
            response_parts.append(f"👤 **Müşteri:** {state['customer_info'].customer_id}")
        
        if state.get("ticket_info"):
            ticket = state["ticket_info"]
            if ticket.ticket_id:
                response_parts.append(f"🎫 **Ticket ID:** {ticket.ticket_id}")
            if ticket.category:
                response_parts.append(f"📋 **Kategori:** {ticket.category}")
            if ticket.priority:
                priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "urgent": "🚨"}
                emoji = priority_emoji.get(ticket.priority.value, "⚪")
                response_parts.append(f"{emoji} **Öncelik:** {ticket.priority.value.upper()}")
        
        response_parts.append("")
        
        # Resolution steps if available
        if state.get("resolution_steps"):
            response_parts.append("✅ **Çözüm Adımları:**")
            for i, step in enumerate(state["resolution_steps"], 1):
                response_parts.append(f"{i}. {step}")
            response_parts.append("")
        
        # Escalation notice
        if state.get("escalation_needed"):
            response_parts.append("⚠️ **Bu talep supervisor'a yönlendirildi**")
            response_parts.append("")
        
        response_parts.append("📞 **Başka bir konuda yardıma ihtiyacınız varsa bize ulaşabilirsiniz.**")
        
        return "\n".join(response_parts)

# Register the Customer Service state and adapter
@register_state(
    name="customer_service",
    description="State management for customer service operations with ticket handling",
    version="1.0.0"
)
class RegisteredCustomerServiceAdapter(CustomerServiceAdapter):
    """Registered Customer Service adapter for automatic discovery"""
    
    @classmethod
    def get_state_class(cls) -> type:
        return CustomerServiceState
