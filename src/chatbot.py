
from typing import List, Tuple
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.config import OPENAI_API_KEY


class Chatbot:
    def __init__(self, llm:LLM):
        self.llm = llm
        self.graph = MainGraph(llm=llm).build_graph()
        self.history = []

    def response_handler(self, history: List[Tuple[str, str]], message: str) -> List[Tuple[str, str]]:
        """
        Gradio 'chat' tipi için uygun yanıt formatı.
        message: kullanıcıdan gelen tek mesaj
        history: [(user_msg: str, bot_msg: str), ...]
        
        Geriye yine aynı formatta [(user_msg, bot_msg)] geçmişini döner.
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
        response = self.graph.invoke(state)

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