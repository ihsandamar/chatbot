# src/graphs/sefim_rag_graph.py

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from langchain.chains.retrieval_qa.base import RetrievalQA
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
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
        self.chroma_db_path = "data/vector_index/sefim_chroma_db"
        self.collection_name = "sefim_master_panel"
        self._initialize_rag_chain()
    
    def _initialize_rag_chain(self):
        """RAG chain'i başlatır"""
        try:
            # Chroma client'ı başlatma
            os.makedirs(self.chroma_db_path, exist_ok=True)
            
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Collection'ı al veya oluştur
            try:
                collection = chroma_client.get_collection(self.collection_name)
                if collection.count() == 0:
                    raise ValueError("Collection boş")
            except:
                print(f"Uyarı: '{self.collection_name}' collection'ı bulunamadı veya boş. Önce dokümanları yükleyin.")
                self.rag_chain = None
                return
            
            # Chroma vectorstore'u oluştur
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=embeddings,
            )
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm.get_chat(),
                retriever=retriever,
                return_source_documents=False
            )
            
            print(f"Chroma RAG chain başarıyla yüklendi. Collection: {self.collection_name}, Döküman sayısı: {collection.count()}")
            
        except Exception as e:
            print(f"Chroma RAG chain başlatılırken hata: {str(e)}")
            self.rag_chain = None
    
    def add_documents_to_chroma(self, texts: List[str], metadatas: List[dict] = None):
        """Yeni dökümanları Chroma'ya ekler"""
        try:
            os.makedirs(self.chroma_db_path, exist_ok=True)
            
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Collection'ı oluştur veya al
            try:
                collection = chroma_client.get_collection(self.collection_name)
            except:
                collection = chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Sefim Master Panel dökümanları"}
                )
            
            # Chroma vectorstore'u oluştur
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=embeddings,
            )
            
            # Dökümanları ekle
            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas or [{"source": "sefim_manual"} for _ in texts]
            )
            
            print(f"Başarıyla {len(texts)} döküman Chroma'ya eklendi. Toplam döküman sayısı: {collection.count()}")
            
            # RAG chain'i yeniden başlat
            self._initialize_rag_chain()
            
        except Exception as e:
            print(f"Döküman eklenirken hata: {str(e)}")
    
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
            error_message = "Bot: Şefim Master Panel dokümanı henüz Chroma veritabanına yüklenmemiş. Lütfen önce dökümanları yükleyin."
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