
# src/core/erp_chatbot.py
from typing import Dict, Any, List, Tuple, Optional
from src.core.chatbot_controller import ERPChatbotController
from src.core.states.base_state import ChatbotState
from src.services.config_loader import ConfigLoader
from src.services.app_logger import log
from langchain_core.messages import HumanMessage, AIMessage

class ERPChatbot:
    """Main ERP Chatbot class with unified interface"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.controller = ERPChatbotController(self.config)
        self.conversation_history = []
        self.logger = log.get(module="erp_chatbot")
        
        self.logger.info("ERP Chatbot initialized", config_path=config_path)
    
    def send_message(self, message: str) -> str:
        """Send a message to the chatbot and get response"""
        try:
            # Create human message
            human_message = HumanMessage(content=message)
            
            # Create or update chatbot state
            current_state = ChatbotState(
                messages=self.conversation_history + [human_message]
            )
            
            # Process message through controller
            response_state = self.controller.process_message(current_state)
            
            # Update conversation history
            self.conversation_history = response_state["messages"]
            
            # Extract and return the latest AI response
            for msg in reversed(response_state["messages"]):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    return msg.content
            
            return "No response generated"
            
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))
            return f"Error processing message: {str(e)}"
    
    def response_handler(self, history: List[Tuple[str, str]], message: str) -> List[Tuple[str, str]]:
        """Gradio-compatible response handler"""
        try:
            # Convert history to messages
            messages = []
            for user_msg, bot_msg in history:
                if user_msg:
                    messages.append(HumanMessage(content=user_msg))
                if bot_msg:
                    messages.append(AIMessage(content=bot_msg))
            
            # Add new user message
            messages.append(HumanMessage(content=message))
            
            # Create state and process
            current_state = ChatbotState(messages=messages)
            response_state = self.controller.process_message(current_state)
            
            # Extract AI response
            ai_response = "No response generated"
            for msg in reversed(response_state["messages"]):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    ai_response = msg.content
                    break
            
            # Update conversation history
            self.conversation_history = response_state["messages"]
            
            # Return updated history
            history.append((message, ai_response))
            return history
            
        except Exception as e:
            self.logger.error("Response handler failed", error=str(e))
            history.append((message, f"Error: {str(e)}"))
            return history
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history reset")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "message_count": len(self.conversation_history),
            "conversation_started": len(self.conversation_history) > 0,
            "last_user_message": self._get_last_user_message(),
            "conversation_context": self._extract_conversation_context()
        }
    
    def _get_last_user_message(self) -> Optional[str]:
        """Get the last user message"""
        for msg in reversed(self.conversation_history):
            if hasattr(msg, 'type') and msg.type == 'human':
                return msg.content
        return None
    
    def _extract_conversation_context(self) -> Dict[str, Any]:
        """Extract conversation context"""
        if not self.conversation_history:
            return {}
        
        return self.message_extractor.extract_conversation_context(self.conversation_history)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "chatbot_info": {
                "version": "1.0.0",
                "conversation_active": len(self.conversation_history) > 0,
                "message_count": len(self.conversation_history)
            },
            "controller_status": self.controller.get_system_status(),
            "health_check": self.controller.health_check()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = ERPChatbot("config/config.yaml")
    
    # Test basic functionality
    print("=== ERP Chatbot Test ===")
    
    # Test 1: General greeting
    response1 = chatbot.send_message("Merhaba")
    print(f"User: Merhaba")
    print(f"Bot: {response1}")
    print()
    
    # Test 2: SQL query intent
    response2 = chatbot.send_message("Müşteriler tablosunu gösterir misin?")
    print(f"User: Müşteriler tablosunu gösterir misin?")
    print(f"Bot: {response2}")
    print()
    
    # Test 3: Customer service intent
    response3 = chatbot.send_message("Sipariş durumumu öğrenmek istiyorum")
    print(f"User: Sipariş durumumu öğrenmek istiyorum")
    print(f"Bot: {response3}")
    print()
    
    # Test 4: System status
    status = chatbot.get_system_info()
    print("=== System Status ===")
    print(f"Overall Health: {status['health_check']['overall_status']}")
    print(f"Active Modules: {list(status['controller_status']['modules'].keys())}")
    print(f"Registered States: {status['controller_status']['state_registry']['total_states']}")
    print()