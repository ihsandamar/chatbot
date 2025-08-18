#!/usr/bin/env python3
"""
Sefim Master Panel PDF dosyasını Chroma veritabanına yükler
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from src.services.config_loader import ConfigLoader

def load_sefim_pdf_to_chroma():
    """PDF dosyasını yükler ve Chroma'ya kaydeder"""
    
    # Config yükle
    config = ConfigLoader.load_config("config/config.yaml")
    
    # PDF dosya yolu
    pdf_path = "data/sefim_master_panel_kullanım.docx"
    
    if not os.path.exists(pdf_path):
        print(f"Hata: {pdf_path} dosyası bulunamadı!")
        return False
    
    print(f"PDF yükleniyor: {pdf_path}")
    
    try:
        # PDF loader
        loader = Docx2txtLoader(pdf_path)
        documents = loader.load()
        
        print(f"PDF başarıyla yüklendi. Sayfa sayısı: {len(documents)}")
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Dökümanları böl
        splits = text_splitter.split_documents(documents)
        print(f"Döküman {len(splits)} parçaya bölündü")
        
        # Embedding modeli
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
        
        # Chroma setup
        chroma_db_path = "data/vector_index/sefim_chroma_db"
        collection_name = "sefim_master_panel"
        
        os.makedirs(chroma_db_path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Eğer collection varsa sil ve yeniden oluştur
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Mevcut '{collection_name}' collection'ı silindi")
        except:
            pass
        
        # Yeni collection oluştur
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Sefim Master Panel kullanım dökümanları"}
        )
        
        # Chroma vectorstore oluştur
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        
        # Dökümanları hazırla
        texts = [doc.page_content for doc in splits]
        metadatas = [
            {
                "source": pdf_path,
                "page": doc.metadata.get("page", 0),
                "chunk_id": i
            }
            for i, doc in enumerate(splits)
        ]
        
        # Dökümanları ekle
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        
        print(f"Basariyla {len(texts)} dokuman parcasi Chroma'ya eklendi!")
        print(f"Collection: {collection_name}")
        print(f"Veritabani yolu: {chroma_db_path}")
        print(f"Toplam dokuman sayisi: {collection.count()}")
        
        return True
        
    except Exception as e:
        print(f"Hata olustu: {str(e)}")
        return False

if __name__ == "__main__":
    print("Sefim PDF -> Chroma yukleme basliyor...")
    success = load_sefim_pdf_to_chroma()
    
    if success:
        print("Islem tamamlandi! Artik Sefim RAG graph'i kullanabilirsiniz.")
    else:
        print("Islem basarisiz!")