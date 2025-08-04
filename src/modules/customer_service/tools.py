
# src/modules/customer_service/tools.py
from langchain_core.tools import tool
from langchain_core.tools.base import BaseToolkit
from typing import Optional, Dict, Any, List
from src.modules.customer_service.state import CustomerInfo, TicketInfo, TicketStatus, TicketPriority
from src.services.app_logger import log
import uuid
from datetime import datetime

@tool
def create_support_ticket(
    customer_id: str,
    title: str,
    description: str,
    category: str = "general_inquiry",
    priority: str = "medium"
) -> Dict[str, Any]:
    """Create a new support ticket for the customer."""
    try:
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        
        ticket_data = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "status": "new",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # In a real implementation, this would save to a database
        log.get().info("Support ticket created", ticket_id=ticket_id, customer_id=customer_id)
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"Support ticket {ticket_id} has been created successfully."
        }
        
    except Exception as e:
        log.get().error("Failed to create support ticket", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create support ticket. Please try again."
        }

@tool
def lookup_customer_info(customer_id: str) -> Dict[str, Any]:
    """Look up customer information by customer ID."""
    try:
        # Mock customer data - in real implementation, query from database
        mock_customers = {
            "CUST001": {
                "customer_id": "CUST001",
                "name": "Ahmet Yılmaz",
                "email": "ahmet.yilmaz@email.com",
                "phone": "+90 532 123 4567",
                "tier": "Premium",
                "registration_date": "2023-01-15",
                "total_orders": 15,
                "last_order_date": "2024-12-01"
            },
            "CUST002": {
                "customer_id": "CUST002", 
                "name": "Fatma Kaya",
                "email": "fatma.kaya@email.com",
                "phone": "+90 533 987 6543",
                "tier": "Standard",
                "registration_date": "2023-06-20",
                "total_orders": 8,
                "last_order_date": "2024-11-28"
            }
        }
        
        customer_data = mock_customers.get(customer_id.upper())
        
        if customer_data:
            log.get().info("Customer info retrieved", customer_id=customer_id)
            return {
                "success": True,
                "customer_data": customer_data
            }
        else:
            return {
                "success": False,
                "message": f"Customer {customer_id} not found in system."
            }
            
    except Exception as e:
        log.get().error("Failed to lookup customer", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to lookup customer information."
        }

@tool
def lookup_order_status(order_id: str) -> Dict[str, Any]:
    """Look up order status by order ID."""
    try:
        # Mock order data
        mock_orders = {
            "ORD001": {
                "order_id": "ORD001",
                "customer_id": "CUST001",
                "status": "delivered",
                "order_date": "2024-11-25",
                "delivery_date": "2024-11-28",
                "total_amount": "₺235.50",
                "items": [
                    {"product": "Kahve Makinesi", "quantity": 1, "price": "₺200.00"},
                    {"product": "Filtre Kağıdı", "quantity": 5, "price": "₺35.50"}
                ],
                "tracking_number": "TRK123456789"
            },
            "ORD002": {
                "order_id": "ORD002",
                "customer_id": "CUST002",
                "status": "in_transit",
                "order_date": "2024-12-01",
                "estimated_delivery": "2024-12-05",
                "total_amount": "₺150.00",
                "items": [
                    {"product": "Espresso Çekirdek", "quantity": 2, "price": "₺150.00"}
                ],
                "tracking_number": "TRK987654321"
            }
        }
        
        order_data = mock_orders.get(order_id.upper())
        
        if order_data:
            log.get().info("Order status retrieved", order_id=order_id)
            return {
                "success": True,
                "order_data": order_data
            }
        else:
            return {
                "success": False,
                "message": f"Order {order_id} not found in system."
            }
            
    except Exception as e:
        log.get().error("Failed to lookup order", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to lookup order status."
        }

@tool
def escalate_to_human(ticket_id: str, reason: str) -> Dict[str, Any]:
    """Escalate ticket to human agent."""
    try:
        escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        
        log.get().info("Ticket escalated to human", 
                      ticket_id=ticket_id, 
                      escalation_id=escalation_id,
                      reason=reason)
        
        return {
            "success": True,
            "escalation_id": escalation_id,
            "message": f"Ticket {ticket_id} has been escalated to human agent. Escalation ID: {escalation_id}",
            "estimated_response_time": "30 minutes"
        }
        
    except Exception as e:
        log.get().error("Failed to escalate ticket", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to escalate ticket to human agent."
        }

@tool
def send_email_notification(customer_email: str, subject: str, message: str) -> Dict[str, Any]:
    """Send email notification to customer."""
    try:
        # Mock email sending - in real implementation, integrate with email service
        notification_id = f"EMAIL-{uuid.uuid4().hex[:8].upper()}"
        
        log.get().info("Email notification sent", 
                      customer_email=customer_email,
                      subject=subject,
                      notification_id=notification_id)
        
        return {
            "success": True,
            "notification_id": notification_id,
            "message": f"Email notification sent to {customer_email}"
        }
        
    except Exception as e:
        log.get().error("Failed to send email", error=str(e))
        return {
            "success": False,  
            "error": str(e),
            "message": "Failed to send email notification."
        }

class CustomerServiceToolkit(BaseToolkit):
    """Toolkit for customer service operations"""
    
    def __init__(self):
        super().__init__(
            name="CustomerServiceToolkit",
            description="Toolkit for customer service operations including ticket management, customer lookup, and escalation"
        )
    
    def get_tools(self) -> List:
        return [
            create_support_ticket,
            lookup_customer_info,
            lookup_order_status,
            escalate_to_human,
            send_email_notification
        ]
