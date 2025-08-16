from langgraph_supervisor import create_supervisor

from src.graphs.chat_graph import ChatGraph
from src.graphs.registry import register_graph
from src.graphs.text2sql_graph import Text2SQLGraph
from src.tools.forza_api_tools import ForzaAPIToolkit
from langchain_core.tools import tool
from datetime import datetime  
from src.graphs.base_graph import BaseGraph

from langgraph.graph import StateGraph, START, END
from src.graphs.base_graph import BaseGraph
from src.models.models import LLM, State
from langgraph.checkpoint.memory import MemorySaver
@tool
def get_today() -> str:
    "Return today"
    return datetime.today().strftime('%Y-%m-%d')

@tool
def get_help_info() -> str:
    """Provide information about available customer service areas and capabilities"""
    return """
🔧 **Customer Service Areas & Capabilities**

📊 **Database & Reporting Services:**
- Product catalog inquiries (search, filter products)
- Sales data analysis and reporting
- Customer data retrieval and management
- Inventory status and stock level checks
- Order history and transaction details
- Performance metrics and analytics

💬 **General Support Services:**
- Product information and specifications
- Account assistance and troubleshooting
- Technical support and guidance
- Process explanations and how-to help
- General questions and conversations
- Feature explanations and tutorials

🎯 **How to Get Help:**
- For data queries: Ask me to "retrieve", "show", "get data from", "generate report"
- For general help: Ask questions, request explanations, or start a conversation
- For service info: Ask "what can you help with?" or "show available services"

Examples:
✅ "Show me all products in electronics category"
✅ "Get sales data for last month"  
✅ "How does the ordering process work?"
✅ "What are the features of product X?"

I'm here to help make your experience smooth and efficient! 🚀
    """



@register_graph("supervisor")
class SupervisorTestGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm, State)





    def build_graph(self):
        chat_graph = ChatGraph(self.llm).build_graph()
        text2sql_graph_agent = Text2SQLGraph(self.llm).build_graph()
        
        # Initialize Forza API Toolkit
        forza_toolkit = ForzaAPIToolkit(base_url="http://localhost:8080")
        forza_tools = forza_toolkit.get_tools()
        
        supervisor = create_supervisor(
            model=self.llm.get_chat(), 
            agents=[chat_graph, text2sql_graph_agent], 
            tools=[get_today, get_help_info] + forza_tools,
            prompt=(
                "You are an intelligent customer service supervisor managing specialized support agents. "
                "Your role is to analyze customer requests and direct them to the most appropriate expert. "
                "\n"
                "Available specialists:\n"
                "- chat_graph: General customer support specialist for conversations, questions, explanations, and assistance\n"
                "- text2sql_graph: Database query specialist for converting customer requests into SQL queries and retrieving data\n"
                "\n"
                "Decision criteria:\n"
                "- Use chat_graph for: General conversations, product information, troubleshooting, explanations, customer service inquiries\n"
                "- Use text2sql_graph for: Data retrieval requests, report generation, database queries, when customer asks to 'get data from', 'show me records', 'query database', 'retrieve information from tables'\n"
                "\n"
                "Always provide helpful, professional customer service. Be proactive in understanding customer needs.\n"
                "\n"
                "Available tools:\n"
                "- get_help_info: Use when customer asks about available services, capabilities, or 'what can you help with?'\n"
                "- get_today: Use when current date is needed\n"
                "- login: Forza ERP user authentication\n"
                "- get_businesses_by_user_id: Get businesses for a specific user\n"
                "- get_branches_by_business_id: Get branches for a specific business\n"
                "- get_user_branches: Complete ERP workflow - login, get businesses, then get all branches (use when customer asks for 'my branches', 'show branches', 'list branches')\n"
                "\n"
                "For ERP operations, use the Forza tools directly when customer requests business data, user login, or branch information.\n"
            )
        )
        memory = MemorySaver()
        return supervisor.compile(checkpointer=memory)                