# Updated main.py with new supervisor system
import argparse
import sys
from src.core.erp_chatbot import ERPChatbot
from src.services.config_loader import ConfigLoader
from src.services.app_logger import log
from src.interfaces.cli_interface import run_cli
from src.interfaces.gradio_interface import run_gradio
from src.interfaces.flask_interface import run_flask

def main():
    """Main entry point for ERP Chatbot with unified state management"""
    parser = argparse.ArgumentParser(description="ERP Chatbot with Advanced State Management")
    parser.add_argument("--interface", choices=["cli", "gradio", "api"], default="gradio",
                        help="Choose interface: cli, gradio, or api")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--test", action="store_true",
                        help="Run system tests")
    parser.add_argument("--health", action="store_true", 
                        help="Show system health status")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = log.get(module="main")
    logger.info("Starting ERP Chatbot", interface=args.interface, config=args.config)
    
    try:
        # Initialize chatbot
        chatbot = ERPChatbot(args.config)
        
        # Handle special commands
        if args.health:
            show_health_status(chatbot)
            return
        
        if args.test:
            run_system_tests(chatbot)
            return
        
        # Run selected interface
        if args.interface == "cli":
            run_cli_interface(chatbot)
        elif args.interface == "gradio":
            run_gradio_interface(chatbot)
        elif args.interface == "api":
            run_flask_interface(chatbot)
            
    except Exception as e:
        logger.error("Failed to start ERP Chatbot", error=str(e))
        print(f"❌ Startup failed: {str(e)}")
        return 1
    
    return 0

def show_health_status(chatbot: ERPChatbot):
    """Display comprehensive system health status"""
    print("🏥 ERP Chatbot System Health Check")
    print("=" * 50)
    
    try:
        system_info = chatbot.get_system_info()
        health = system_info["health_check"]
        
        # Overall status
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "error": "❌"}
        print(f"Overall Status: {status_emoji.get(health['overall_status'], '❓')} {health['overall_status'].upper()}")
        print()
        
        # Component status
        print("📋 Component Status:")
        for component, details in health["components"].items():
            status = details["status"]
            emoji = status_emoji.get(status, "❓")
            print(f"  {emoji} {component.replace('_', ' ').title()}: {status}")
            if "details" in details:
                print(f"    ℹ️  {details['details']}")
            if "error" in details:
                print(f"    ❌ {details['error']}")
        print()
        
        # Module information
        modules = system_info["controller_status"]["modules"]
        print("🧩 Available Modules:")
        for module_name, module_info in modules.items():
            status = "✅" if module_info["status"] == "active" else "❌"
            print(f"  {status} {module_name.title()}")
            if module_info["status"] == "active" and "info" in module_info:
                info = module_info["info"]
                print(f"    📝 {info.get('description', 'No description')}")
                print(f"    🔢 Version: {info.get('version', 'Unknown')}")
        print()
        
        # State registry
        registry = system_info["controller_status"]["state_registry"]
        print("🗂️  State Registry:")
        print(f"  📊 Total registered states: {registry['total_states']}")
        print(f"  📋 States: {', '.join(registry['registered_states'])}")
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")

def run_system_tests(chatbot: ERPChatbot):
    """Run comprehensive system tests"""
    print("🧪 Running ERP Chatbot System Tests")
    print("=" * 50)
    
    try:
        # Test 1: Basic message processing
        print("📨 Test 1: Basic Message Processing")
        test_messages = [
            "Merhaba",
            "Bu ayın ciro raporunu göster",
            "Müşteriler tablosunu listele",
            "Sipariş durumumu öğrenmek istiyorum",
            "Sistem hatası alıyorum"
        ]
        
        for i, message in enumerate(test_messages, 1):
            try:
                response = chatbot.send_message(message)
                status = "✅" if response and "Error" not in response else "❌"
                print(f"  {status} Test {i}: {message[:30]}...")
            except Exception as e:
                print(f"  ❌ Test {i}: {message[:30]}... -> Error: {str(e)}")
        print()
        
        # Test 2: State management
        print("🔄 Test 2: State Management")
        try:
            conversation_summary = chatbot.get_conversation_summary()
            status = "✅" if conversation_summary["conversation_started"] else "⚠️"
            print(f"  {status} Conversation tracking: {conversation_summary['message_count']} messages")
            print(f"  📝 Last message: {conversation_summary.get('last_user_message', 'None')[:50]}...")
        except Exception as e:
            print(f"  ❌ State management test failed: {str(e)}")
        print()
        
        # Test 3: Module integration
        print("🧩 Test 3: Module Integration")
        system_info = chatbot.get_system_info()
        modules = system_info["controller_status"]["modules"]
        
        for module_name, module_info in modules.items():
            status = "✅" if module_info["status"] == "active" else "❌"
            print(f"  {status} {module_name.title()} module: {module_info['status']}")
        print()
        
        # Test 4: Supervisor graph (if available)
        print("🎯 Test 4: Supervisor Graph")
        try:
            # Test supervisor graph specifically
            from src.graphs.supervisor_graph import TestSupervisorGraph
            supervisor_tests = TestSupervisorGraph()
            
            # Run intent detection test
            intent_results = supervisor_tests.test_intent_detection_accuracy()
            passed = sum(1 for r in intent_results if r["passed"])
            total = len(intent_results)
            print(f"  📊 Intent detection: {passed}/{total} tests passed")
            
            # Run routing test
            routing_result = supervisor_tests.test_routing_logic_correctness()
            status = "✅" if routing_result["test_passed"] else "❌"
            print(f"  {status} Routing logic test")
            
            # Run error handling test
            error_result = supervisor_tests.test_error_handling_robustness()
            status = "✅" if error_result["test_passed"] else "❌"
            print(f"  {status} Error handling test")
            
        except ImportError:
            print("  ⚠️  Supervisor graph tests not available")
        except Exception as e:
            print(f"  ❌ Supervisor graph tests failed: {str(e)}")
        
        print("\n🏁 System tests completed!")
        
    except Exception as e:
        print(f"❌ System tests failed: {str(e)}")

def run_cli_interface(chatbot: ERPChatbot):
    """Run CLI interface with enhanced functionality"""
    print("🖥️  ERP Chatbot CLI Interface")
    print("=" * 40)
    print("Commands:")
    print("  /help     - Show help")
    print("  /status   - Show system status")
    print("  /reset    - Reset conversation")
    print("  /quit     - Exit")
    print("=" * 40)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == '/help':
                print_cli_help()
                continue
            elif user_input.lower() == '/status':
                show_health_status(chatbot)
                continue
            elif user_input.lower() == '/reset':
                chatbot.reset_conversation()
                print("🔄 Conversation reset!")
                continue
            
            # Process normal message
            response = chatbot.send_message(user_input)
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def print_cli_help():
    """Print CLI help information"""
    print("""
📖 ERP Chatbot Help

🎯 What I can do:
  📊 Generate reports and analytics
  🔍 Execute database queries  
  👥 Handle customer service requests
  🎧 Provide technical support
  💡 Process feature requests
  📚 Access documentation
  🏢 Provide company information

💬 Example requests:
  "Bu ayın ciro raporunu göster"
  "Müşteriler tablosunu listele"
  "Sipariş durumumu öğrenmek istiyorum"
  "Sistem hatası alıyorum, yardım lazım"
  "Yeni rapor özelliği eklenebilir mi?"

⚙️  Special commands:
  /help   - Show this help
  /status - System health check
  /reset  - Reset conversation
  /quit   - Exit chatbot
""")

def run_gradio_interface(chatbot: ERPChatbot):
    """Run Gradio interface with state management"""
    try:
        import gradio as gr
        
        def chat_interface(history, message):
            """Enhanced Gradio chat interface"""
            return chatbot.response_handler(history, message)
        
        def get_system_status():
            """Get system status for display"""
            try:
                info = chatbot.get_system_info()
                health = info["health_check"]
                
                status_text = f"""
**System Status:** {health['overall_status'].upper()}

**Active Modules:**
{chr(10).join([f"• {name.title()}: {info['status']}" for name, info in info['controller_status']['modules'].items()])}

**Message Count:** {info['chatbot_info']['message_count']}
**Conversation Active:** {info['chatbot_info']['conversation_active']}
"""
                return status_text
            except Exception as e:
                return f"Status check failed: {str(e)}"
        
        def reset_conversation():
            """Reset conversation and return status"""
            chatbot.reset_conversation()
            return "Conversation reset successfully!"
        
        # Create Gradio interface
        with gr.Blocks(title="ERP Chatbot - Advanced State Management") as demo:
            gr.Markdown("# 🏢 ERP Chatbot with Unified State Management")
            gr.Markdown("Advanced AI assistant for ERP operations with intelligent module routing")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot_ui = gr.Chatbot(
                        label="Chat with ERP Assistant",
                        height=500,
                        placeholder="Merhaba! Size nasıl yardımcı olabilirim?"
                    )
                    
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### System Status")
                    status_display = gr.Textbox(
                        label="Status",
                        value=get_system_status(),
                        lines=10,
                        interactive=False
                    )
                    
                    refresh_btn = gr.Button("Refresh Status")
                    reset_btn = gr.Button("Reset Conversation", variant="stop")
                    
                    gr.Markdown("### Quick Actions")
                    gr.Examples(
                        examples=[
                            "Merhaba",
                            "Bu ayın ciro raporunu göster",
                            "Müşteriler tablosunu listele",
                            "Sipariş durumumu öğrenmek istiyorum",
                            "Sistem hatası alıyorum",
                            "Yardım"
                        ],
                        inputs=msg_input
                    )
            
            # Event handlers
            send_btn.click(
                chat_interface,
                inputs=[chatbot_ui, msg_input],
                outputs=[chatbot_ui]
            ).then(
                lambda: "",
                outputs=[msg_input]
            )
            
            msg_input.submit(
                chat_interface,
                inputs=[chatbot_ui, msg_input],
                outputs=[chatbot_ui]
            ).then(
                lambda: "",
                outputs=[msg_input]
            )
            
            clear_btn.click(
                lambda: [],
                outputs=[chatbot_ui]
            )
            
            refresh_btn.click(
                get_system_status,
                outputs=[status_display]
            )
            
            reset_btn.click(
                reset_conversation,
                outputs=[status_display]
            )
        
        print("🌐 Starting Gradio interface...")
        demo.launch(server_name="0.0.0.0", share=False, debug=False)
        
    except ImportError:
        print("❌ Gradio not installed. Run: pip install gradio")
    except Exception as e:
        print(f"❌ Gradio interface failed: {str(e)}")

def run_flask_interface(chatbot: ERPChatbot):
    """Run Flask API interface with state management"""
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route("/chat", methods=["POST"])
        def chat_endpoint():
            """Chat endpoint with state management"""
            try:
                data = request.get_json()
                message = data.get("message", "")
                
                if not message:
                    return jsonify({"error": "Message is required"}), 400
                
                response = chatbot.send_message(message)
                
                return jsonify({
                    "response": response,
                    "status": "success",
                    "timestamp": chatbot.get_conversation_summary().get("conversation_context", {}).get("extraction_timestamp")
                })
                
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "status": "error"
                }), 500
        
        @app.route("/status", methods=["GET"])
        def status_endpoint():
            """System status endpoint"""
            try:
                system_info = chatbot.get_system_info()
                return jsonify(system_info)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @app.route("/health", methods=["GET"])
        def health_endpoint():
            """Health check endpoint"""
            try:
                health = chatbot.get_system_info()["health_check"]
                return jsonify(health)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @app.route("/reset", methods=["POST"])
        def reset_endpoint():
            """Reset conversation endpoint"""
            try:
                chatbot.reset_conversation()
                return jsonify({"message": "Conversation reset successfully"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @app.route("/", methods=["GET"])
        def index():
            """API documentation"""
            return jsonify({
                "name": "ERP Chatbot API",
                "version": "1.0.0",
                "description": "Advanced ERP chatbot with unified state management",
                "endpoints": {
                    "POST /chat": "Send message to chatbot",
                    "GET /status": "Get system status",
                    "GET /health": "Get health check",
                    "POST /reset": "Reset conversation",
                    "GET /": "This documentation"
                },
                "example_request": {
                    "url": "/chat",
                    "method": "POST",
                    "body": {"message": "Bu ayın ciro raporunu göster"}
                }
            })
        
        print("🚀 Starting Flask API server...")
        print("📡 API endpoints:")
        print("  POST /chat - Send message")
        print("  GET /status - System status")
        print("  GET /health - Health check")
        print("  POST /reset - Reset conversation")
        print("  GET / - API documentation")
        print("\n🌐 Server running on http://localhost:5000")
        
        app.run(host="0.0.0.0", port=5000, debug=False)
        
    except ImportError:
        print("❌ Flask not installed. Run: pip install flask")
    except Exception as e:
        print(f"❌ Flask interface failed: {str(e)}")

if __name__ == "__main__":
    exit(main())

# Example configuration files to support the new system

# config/supervisor_config.yaml
supervisor_config = """
llm:
  model: gpt-4o-mini
  temperature: 0.0
  api_key: ${OPENAI_API_KEY}

database:
  uri: ${DATABASE_URI}
  pool_size: 5
  max_overflow: 10

chatbot:
  graph_type: supervisor
  thread_id: "supervisor_session"

supervisor:
  intent_detection:
    confidence_threshold: 0.7
    clarification_threshold: 0.4
    fallback_to_options: true
  
  modules:
    enabled:
      - text2sql
      - customer_service
      - reporting
      - support
      - documents
      - request
      - company_info
    
    routing_preferences:
      high_confidence_direct_route: true
      low_confidence_clarification: true
      show_confidence_scores: false
  
  conversation:
    welcome_new_users: true
    remember_context: true
    max_history_length: 50
    session_timeout: 3600

logging:
  level: INFO
  log_to_file: true
  log_dir: logs
"""

# config/docker-compose.yml for deployment
docker_compose = """
version: '3.8'

services:
  erp-chatbot:
    build: .
    ports:
      - "5000:5000"
      - "7860:7860"  # Gradio
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URI=${DATABASE_URI}
      - APP_ENV=prod
      - LOG_TO_FILE=1
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  database:
    image: mcr.microsoft.com/mssql/server:2019-latest
    environment:
      - ACCEPT_EULA=Y
      - SA_PASSWORD=${SA_PASSWORD}
      - MSSQL_PID=Express
    ports:
      - "1433:1433"
    volumes:
      - sqlserver_data:/var/opt/mssql
    restart: unless-stopped

volumes:
  sqlserver_data:
"""

# Dockerfile
dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    unixodbc \\
    unixodbc-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \\
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \\
    && apt-get update \\
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Expose ports
EXPOSE 5000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "main.py", "--interface", "api"]
"""

# Updated requirements.txt
requirements = """
Flask==3.1.1
gradio==5.31.0
langchain_community==0.3.24
langchain_core==0.3.63
langchain_openai==0.3.18
langgraph==0.4.7
pydantic==2.11.5
python-dotenv==1.1.0
PyYAML==6.0.2
rich==13.7.0
structlog==23.2.0
sqlalchemy==2.0.23
pyodbc==5.0.1
asyncio==3.4.3
typing-extensions==4.8.0
"""

# Save example files function
def save_example_files():
    """Save example configuration files"""
    import os
    from pathlib import Path
    
    # Create directories
    Path("config").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Save configuration files
    files_to_save = {
        "config/supervisor_config.yaml": supervisor_config,
        "docker-compose.yml": docker_compose,
        "Dockerfile": dockerfile_content,
        "requirements.txt": requirements
    }
    
    for file_path, content in files_to_save.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"📁 Created: {file_path}")
    
    print("\n✅ Example configuration files created!")
    print("\n🚀 Quick start:")
    print("1. Set environment variables in config/.env")
    print("2. Run: python main.py --interface gradio")
    print("3. Or run: docker-compose up")

# Additional utility: Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and usage"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "module_usage": {},
            "intent_accuracy": {}
        }
        self.logger = log.get(module="performance_monitor")
    
    def record_request(self, success: bool, response_time: float, module: str = None, intent: str = None):
        """Record request metrics"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average response time
        current_avg = self.metrics["average_response_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["average_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
        
        # Track module usage
        if module:
            self.metrics["module_usage"][module] = self.metrics["module_usage"].get(module, 0) + 1
        
        # Track intent accuracy
        if intent:
            self.metrics["intent_accuracy"][intent] = self.metrics["intent_accuracy"].get(intent, 0) + 1
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        success_rate = (self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1)) * 100
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "failure_rate": 100 - success_rate
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "module_usage": {},
            "intent_accuracy": {}
        }
        self.logger.info("Performance metrics reset")

# Save example files when module is run directly
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--save-examples":
        save_example_files()
    else:
        main()