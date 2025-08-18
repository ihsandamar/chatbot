# src/graphs/sefim_rag_graph.py

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS, C
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


# Embedding modeli tanımı
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=config.llm.api_key,
    base_url=config.llm.base_url,
)

@register_graph("sefim_rag")
class SefimRAGGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm, State)
        self.vectorstore_path = "data/vector_index/sefim_master_panel_index"
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
            print(f"Uyarı: {self.vectorstore_path} bulunamadı. Önce sefim_master_panel_kullanım.pdf dosyasını vector index'e yükleyin.")
            self.rag_chain = None
    
    def run_sefim_rag(self, state: State) -> State:
        """Sefim RAG node fonksiyonu"""
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
            error_message = "Bot: Şefim Master Panel dokümanı henüz yüklenmemiş. Lütfen önce sefim_master_panel_kullanım.pdf dosyasını sisteme yükleyin."
            state["messages"] = state.get("messages", []) + [error_message]
            return state

        # Şefim Master Panel'e özel sistem yönlendirmesi
        system_prompt = """
Sen bir Şefim Master Panel kullanım destek uzmanısın. Aşağıdaki kullanıcı sorusu, Şefim Master Panel programının kullanımı hakkında olabilir.
Özellikle aşağıdaki konularla ilgili cevap vermeye odaklan:

- Şefim Master Panel arayüzü kullanımı
- Menü ve butonların işlevleri
- Master veri yönetimi (müşteriler, ürünler, kategoriler vb.)
- Raporlama özellikleri
- Arama ve filtreleme işlemleri
- Veri girişi ve düzenleme işlemleri
- Kullanıcı ayarları ve konfigürasyon
- Sistem entegrasyonları
- Sorun giderme ve ipuçları
- İş akışları ve süreçler

Cevabında:
- Master Panel dokümanındaki bilgileri referans al
- Adım adım talimatlar ver
- Eğer dokümanda ekran görüntüleri varsa bunlara atıfta bulun
- Net, anlaşılır ve kullanıcı dostu bir tonla cevap ver
- Gerekirse menü yollarını belirt (örn: Ana Menü > Masterlar > Müşteri Tanımları)
- Pratik örnekler ver

Kullanıcı sorusu:
"""
        
        prompt = system_prompt.strip() + "\n" + user_input
        result = self.rag_chain.run(prompt)
        response = f"Bot: {result}"

        state["messages"] = state.get("messages", []) + [response]
        return state

    def build_graph(self):
        """Graph'ı oluşturur"""
        print("Sefim RAG graph çalıştı...")
        graph = StateGraph(State)
        graph.add_node("sefim_rag", RunnableLambda(self.run_sefim_rag))
        graph.set_entry_point("sefim_rag")
        graph.set_finish_point("sefim_rag")
        return graph.compile()