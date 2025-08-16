# src/graphs/forza_rag_graph.py

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import TypedDict
from typing import List, Optional, TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM, State
from src.services.config_loader import ConfigLoader

config = ConfigLoader.load_config("config/config.yaml")

load_dotenv()


class AgentState(TypedDict):
    user_prompt: str
    refined_text: str
    missing_fields: list[str]
    filled_fields: dict[str, str]
    latest_answer: str
    result: str
    search_results: str
    input: str
    refined_prompt: str
    output: str
    followup_question: str
    stop: bool
    intent: str
    sql_query: str
    retrieved_docs: list[str]
    awaiting_followup: bool
    messages: Annotated[list[AnyMessage], add_messages]


# Embedding modeli tanımı
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=config.llm.api_key,
    base_url=config.llm.base_url,
)
@register_graph("forza_rag")
class ForzaRAGGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm, State)
        self.vectorstore_path = "data/vector_index/forza_kurulum_index"
        self._initialize_rag_chain()
    
    def _initialize_rag_chain(self):
        """RAG chain'i başlatır"""
        try:
            vectorstore = FAISS.load_local(self.vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm.get_chat(),
                retriever=retriever,
                return_source_documents=False
            )
        except FileNotFoundError:
            print(f"Uyarı: {self.vectorstore_path} bulunamadı. Önce forza_kurulum.pdf dosyasını vector index'e yükleyin.")
            self.rag_chain = None
    
    def run_forza_rag(self, state: State) -> State:
        """Forza RAG node fonksiyonu"""
        # Try to get user input from various possible keys
        user_input = state.get("user_query", "")
        
        # If no user_query, extract from latest human message
        if not user_input:
            messages = state.get("messages", [])
            for message in reversed(messages):
                if hasattr(message, 'type') and message.type == 'human':
                    # Handle different message content formats
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            for content_part in message.content:
                                if isinstance(content_part, dict) and content_part.get('type') == 'text':
                                    user_input = content_part.get('text', '')
                                    break
                        else:
                            user_input = message.content
                        break
        
        if not user_input:
            raise ValueError("Giriş verisi eksik: 'user_query' veya 'messages' bulunamadı.")

        if self.rag_chain is None:
            error_message = "Bot: Forza kurulum dokümanı henüz yüklenmemiş. Lütfen önce forza_kurulum.pdf dosyasını sisteme yükleyin."
            state["messages"] = state.get("messages", []) + [error_message]
            return state

        # Forza kurulum dokümanına özel sistem yönlendirmesi
        system_prompt = """
Sen bir Forza programı kurulum destek uzmanısın. Aşağıdaki kullanıcı sorusu, Forza programının kurulumu hakkında olabilir.
Özellikle aşağıdaki konularla ilgili cevap vermeye odaklan:

- Forza programı kurulum adımları
- Sistem gereksinimleri
- Kurulum sorunları ve çözümleri  
- Lisans aktivasyonu
- Kurulum sonrası yapılandırma
- Güncelleme işlemleri
- Kaldırma işlemleri
- Teknik sorun giderme

Cevabında:
- Kurulum dokümanındaki bilgileri referans al
- Adım adım talimatlar ver
- Eğer dokümanda resimler/ekran görüntüleri varsa bunlara atıfta bulun
- Net, anlaşılır ve teknik destek tonuyla cevap ver
- Gerekirse örnek ver veya uyarılarda bulun

Kullanıcı sorusu:
"""
        
        prompt = system_prompt.strip() + "\n" + user_input
        result = self.rag_chain.run(prompt)
        response = f"Bot: {result}"

        state["messages"] = state.get("messages", []) + [response]
        return state

    def build_graph(self):
        """Graph'ı oluşturur"""
        print("Forza RAG graph çalıştı...")
        graph = StateGraph(State)
        graph.add_node("forza_rag", RunnableLambda(self.run_forza_rag))
        graph.set_entry_point("forza_rag")
        graph.set_finish_point("forza_rag")
        return graph.compile()