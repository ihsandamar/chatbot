
from typing import List, Tuple
from src.graphs.base_graph import BaseGraph
from src.graphs.main_graph import MainGraph
from src.models.models import LLM


class Chatbot:
    def __init__(self, llm:LLM, graph: BaseGraph, config: dict = None):
        self.llm = llm
        self.graph = graph
        self.history = []
        self.config = config if config else {}

    def response_handler(self, history: List[Tuple[str, str]], message: str) -> List[Tuple[str, str]]:
        """
        Response handler for Gradio 'chat' type.
        message: single message from the user
        history: [(user_msg: str, bot_msg: str), ...]
        Returns the history in the same format [(user_msg, bot_msg)].
        """

        # LangGraph'e uygun mesaj listesi oluştur
        messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        # Yeni gelen kullanıcı mesajını da ekle

        messages.append({"role": "user", "content": message})

        # State hazırla ve invoke et
        state = {"messages": messages}
        response = self.graph.invoke(state, config=self.config)

        # Asistan cevabını al
        if "messages" in response and isinstance(response["messages"], list):
            last = response["messages"][-1]
            content = last.content if hasattr(last, "content") else last.get("content", "")
            # Yeni çifti geçmişe ekleyip döndür
            history.append((message, content))
            return history

        # Hata varsa kullanıcıya mesaj döndür
        history.append((message, "Bir hata oluştu, lütfen tekrar deneyin."))
        return history
    
    def send(self, state: dict) -> str:
        """
        Tüm arayüzlerde kullanılabilecek sade mesaj gönderici fonksiyonu.
        Girdi: {"messages": [{"role": "user", "content": "..."}]}
        Çıktı: string cevap
        """
        response = self.graph.invoke(state, config=self.config)
        print(response.tool_calls)


        if "messages" in response and isinstance(response["messages"], list):
            last = response["messages"][-1]
            return last.content if hasattr(last, "content") else last.get("content", "")

        return "Bir hata oluştu, lütfen tekrar deneyin."