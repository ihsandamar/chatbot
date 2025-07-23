from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from src.services.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader().load_config()
console = Console()

# Define the shared state schema
class State(TypedDict):
    messages: List[HumanMessage | AIMessage]
    current_graph: str  # Graf durumunu takip etmek için eklendi

# Global state to track current graph
class GraphManager:
    def __init__(self):
        self.current_graph = "first"
        self.state: State = {'messages': [], 'current_graph': 'first'}
    
    def switch_to_second_graph(self):
        self.current_graph = "second"
        self.state['current_graph'] = "second"
        console.print(Panel.fit(
            ">>> Handoff gerçekleşti: İkinci grafa geçiliyor!", style="bold yellow"
        ))

# Build the unified graph that handles both states
def build_unified_graph(llm: ChatOpenAI, graph_manager: GraphManager) -> StateGraph:
    graph = StateGraph(State)
    
    def router_node(state: State) -> State:
        """Hangi graph'ın aktif olduğunu kontrol eder"""
        current_graph = state.get('current_graph', 'first')
        
        if current_graph == 'first':
            return first_graph_logic(state, llm, graph_manager)
        else:
            return second_graph_logic(state, llm)
    
    def first_graph_logic(state: State, llm: ChatOpenAI, graph_manager: GraphManager) -> State:
        """İlk graf mantığı"""
        last_message = state['messages'][-1]
        
        # "devret" kontrolü
        if isinstance(last_message, HumanMessage) and 'devret' in last_message.content.lower():
            # Graf değişikliği
            graph_manager.switch_to_second_graph()
            state['current_graph'] = 'second'
            
            # İkinci grafa geçiş mesajı
            state['messages'].append(
                AIMessage(content="[System] İkinci graf devralındı. Artık ikinci grafta çalışıyorsunuz.")
            )
            return state
        
        # Normal ilk graf işlemi
        system_msg = AIMessage(content="[System] İlk graf çalışıyor. 'devret' yazarak ikinci grafa geçebilirsin.")
        state['messages'].append(system_msg)
        
        # LLM yanıtı
        response = llm.invoke(state['messages'])
        state['messages'].append(response)
        
        return state
    
    def second_graph_logic(state: State, llm: ChatOpenAI) -> State:
        """İkinci graf mantığı"""
        # İkinci graf sistem mesajı (sadece ilk geçişte)
        last_messages = state['messages'][-2:] if len(state['messages']) >= 2 else state['messages']
        
        # Eğer son mesaj sistem mesajı değilse, sistem mesajı ekle
        if not any("[System] İkinci graf" in msg.content for msg in last_messages if isinstance(msg, AIMessage)):
            system_msg = AIMessage(content="[System] İkinci graf aktif. Burada devam edebilirsin.")
            state['messages'].append(system_msg)
        
        # LLM yanıtı
        response = llm.invoke(state['messages'])
        state['messages'].append(response)
        
        return state
    
    # Graf düğümlerini ekle
    graph.add_node('router', router_node)
    graph.add_edge(START, 'router')
    graph.add_edge('router', END)
    
    return graph.compile()

# Interactive demo runner with rich console UI
def run_demo():
    # Initialize LLM and graph manager
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        api_key=config.llm.api_key
    )
    
    graph_manager = GraphManager()
    unified_graph = build_unified_graph(llm, graph_manager)

    console.print(Panel.fit("Handoff Demo Başladı", style="bold green"))
    console.print("Mesajınızı yazın ('exit' ile çıkış, 'devret' ile ikinci grafa geç):")
    
    while True:
        # Mevcut graf durumunu göster
        current_status = "İlk Graf" if graph_manager.current_graph == "first" else "İkinci Graf"
        user_input = console.input(f"[bold cyan]User ({current_status})[/bold cyan]: ")
        
        if user_input.strip().lower() == 'exit':
            console.print(Panel.fit("Demo sonlandırıldı.", style="bold red"))
            break
        
        # Kullanıcı mesajını state'e ekle
        graph_manager.state['messages'].append(HumanMessage(content=user_input))
        
        # Graf'ı çalıştır
        result_state = unified_graph.invoke(graph_manager.state)
        
        # State'i güncelle
        graph_manager.state = result_state
        
        # Son AI mesajını yazdır
        if graph_manager.state['messages']:
            last_ai_msg = None
            # Sondan başlayarak son AI mesajını bul
            for msg in reversed(graph_manager.state['messages']):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg
                    break
            
            if last_ai_msg:
                console.print(f"[bold magenta]Assistant[/bold magenta]: {last_ai_msg.content}\n")

if __name__ == '__main__':
    run_demo()