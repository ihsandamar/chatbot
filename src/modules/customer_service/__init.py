
# src/modules/customer_service/__init__.py
"""
Customer Service Module

This module provides comprehensive customer service capabilities
with advanced state management, ticket handling, and escalation features.
"""

from .state import (
    CustomerServiceState, 
    CustomerServiceAdapter, 
    RegisteredCustomerServiceAdapter,
    CustomerInfo,
    TicketInfo,
    TicketPriority,
    TicketStatus
)
from .tools import CustomerServiceToolkit
from .integration import CustomerServiceModule

__all__ = [
    'CustomerServiceState',
    'CustomerServiceAdapter',
    'RegisteredCustomerServiceAdapter', 
    'CustomerInfo',
    'TicketInfo',
    'TicketPriority',
    'TicketStatus',
    'CustomerServiceToolkit',
    'CustomerServiceModule'
]