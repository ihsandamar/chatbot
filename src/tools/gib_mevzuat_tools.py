"""
GIB Mevzuat API Tools
Gelir İdaresi Başkanlığı mevzuat araştırma araçları
"""

import logging
import requests
import json
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import re


class GIBMevzuatSearchTool(BaseTool):
    """GIB mevzuat arama aracı"""
    
    name: str = "gib_mevzuat_search"
    description: str = """
    Gelir İdaresi Başkanlığı mevzuat veritabanında arama yapar.
    Kullanıcının belirttiği konu ile ilgili mevzuat, genelge, tebliğ vb. bulur.
    Arama terimleri: kullanıcının sorusundaki anahtar kelimeler
    Tür: madde, gerekce, bkk, cbk, teblig, yonetmelikler, genelYazilar, icGenelge, ozelge, sirkuler
    """
    
    def _run(self, search_terms: str, mevzuat_type: str = "genelYazilar") -> str:
        """GIB mevzuat arama"""
        try:
            # API URL
            base_url = "https://gib.gov.tr/api/gibportal/mevzuat"
            url = f"{base_url}/{mevzuat_type}/list"
            
            # Arama parametreleri
            params = {
                "page": 0,
                "size": 50,  # Daha fazla sonuç al
                "sortFieldName": "priority",
                "sortType": "ASC"
            }
            
            # POST data - hem title hem de description'da ara
            search_data = {
                "kanunType": 1,
                "title": search_terms,
                "kanunNo": "",
                "description": search_terms,  # Açıklamada da ara
                "ktype": 1
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            # API çağrısı
            response = requests.post(url, 
                                   params=params, 
                                   json=search_data, 
                                   headers=headers,
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == 200 and data.get("resultContainer"):
                    content = data["resultContainer"].get("content", [])
                    
                    if content:
                        results = []
                        for item in content[:15]:  # Daha fazla sonuç al
                            result = {
                                "title": item.get("title", ""),
                                "description": item.get("description", "")[:1000] + "..." if len(item.get("description", "")) > 1000 else item.get("description", ""),
                                "siteLink": item.get("siteLink", ""),
                                "tarih": item.get("tarih", ""),
                                "kanunTitle": item.get("kanunTitle", ""),
                                "kanunNo": item.get("kanunNo", ""),
                                "id": item.get("id", ""),  # Unique ID ekle
                                "url": item.get("url", "")  # URL ekle
                            }
                            results.append(result)
                        
                        return json.dumps(results, ensure_ascii=False, indent=2)
                    else:
                        return f"'{search_terms}' için {mevzuat_type} türünde sonuç bulunamadı."
                else:
                    messages = data.get('messages', [])
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        message_text = messages[0].get('text', 'Bilinmeyen hata')
                        return f"API mesajı: {message_text}"
                    return f"'{search_terms}' için {mevzuat_type} türünde sonuç bulunamadı."
            else:
                return f"API hatası: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Arama hatası: {str(e)}"


class GIBContentFetchTool(BaseTool):
    """GIB mevzuat içerik alma aracı"""
    
    name: str = "gib_content_fetch"
    description: str = """
    GIB mevzuat linkinden tam içeriği alır.
    HTML içeriğini temizleyip okunabilir metin haline getirir.
    Link: GIB mevzuat sitesindeki tam URL
    """
    
    def _run(self, site_link: str) -> str:
        """Mevzuat içeriğini getir"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(site_link, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # HTML'i parse et
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # İçerik alanını bul
                content_div = soup.find('div', class_='content') or soup.find('div', class_='mevzuat-content') or soup.find('div', id='content')
                
                if content_div:
                    # Script ve style taglerini kaldır
                    for script in content_div(["script", "style"]):
                        script.decompose()
                    
                    # Metni al ve temizle
                    text = content_div.get_text()
                    
                    # Fazla boşlukları temizle
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text = re.sub(r' +', ' ', text)
                    text = text.strip()
                    
                    # İlk 5000 karakteri al
                    if len(text) > 5000:
                        text = text[:5000] + "\n\n[İçerik kısaltıldı...]"
                    
                    return text
                else:
                    # Genel body içeriğini al
                    body = soup.find('body')
                    if body:
                        text = body.get_text()
                        text = re.sub(r'\n\s*\n', '\n\n', text)
                        text = re.sub(r' +', ' ', text)
                        text = text.strip()
                        
                        if len(text) > 3000:
                            text = text[:3000] + "\n\n[İçerik kısaltıldı...]"
                        
                        return text
                    else:
                        return "İçerik bulunamadı."
            else:
                return f"Sayfa alınamadı: HTTP {response.status_code}"
                
        except Exception as e:
            return f"İçerik alma hatası: {str(e)}"


class GIBKeywordExtractorTool(BaseTool):
    """GIB mevzuat araması için AI-driven anahtar kelime çıkarma aracı"""
    
    name: str = "extract_gib_keywords"
    description: str = """
    Kullanıcının vergi/mali mevzuat sorusundan GIB özelge araması için en uygun anahtar kelimeleri çıkarır.
    Tamamen AI-driven yaklaşım kullanarak Türk vergi terminolojisine özel optimizasyon yapar.
    """

    llm: Any = Field(default=None, exclude=True)
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'llm', llm)
    
    def _run(self, user_query: str) -> str:
        """AI-driven anahtar kelime çıkarma"""
        try:
            print(f"GIB anahtar kelime çıkarma başlatıldı: {user_query[:50]}")
            
            if not self.llm:
                return self._emergency_fallback(user_query)
            
            # Tek prompt ile tüm işi yap
            result = self._extract_with_ai(user_query)
            
            print(f"Anahtar kelime çıkarma tamamlandı")
            return result
            
        except Exception as e:
            print(f"Anahtar kelime çıkarma hatası: {str(e)}")
            return self._emergency_fallback(user_query)
    
    def _extract_with_ai(self, user_query: str) -> str:
        """AI ile anahtar kelime çıkarma"""
        try:
            prompt = f"""Sen Türkiye vergi mevzuatı uzmanısın. GIB mevzuat araması için anahtar kelime çıkarıyorsun.

Kullanıcı sorusu: "{user_query}"

GÖREV: Bu sorudan GIB mevzuat araması için en uygun 3-5 anahtar kelimeyi çıkar.

ÖNEMLİ KURALLAR:
1. **Vergi terminolojisi öncelikli**: KDV, ÖTV, gelir vergisi, kurumlar vergisi, stopaj, tevkifat
2. **Teknik terimler önemli**: 
   - POS cihazı = Ödeme Kaydedici Cihaz = ÖKC
   - Stopaj = Tevkifat = Vergi Kesintisi
   - İade = Geri Ödeme = Mahsup
3. **Birleşik terimleri koru**: "ödeme kaydedici cihaz", "teknoloji geliştirme bölgesi"
4. **Eş anlamlıları ekle**: POS için hem "POS cihazı" hem "ÖKC" ekle
5. **Meslek/sektör tespiti**:
   - Avukat, doktor, noter → "Serbest Meslek" ekle
   - İnternet, online → "E-ticaret" ekle  
   - Kira, emlak → "Gayrimenkul" ekle
   - Akaryakıt, benzinlik → "Akaryakıt İstasyonu" ekle
6. **Stop words çıkar**: ve, ile, için, hakkında, nasıl, nedir, olan, yapılan

ÖRNEKLER:
- "ödeme kaydedici cihaz kayıt işlemleri" → ["ödeme kaydedici cihaz", "ÖKC", "POS cihazı", "kayıt"]
- "avukat pos cihazı kullanımı" → ["avukat", "serbest meslek", "POS cihazı", "ÖKC"]
- "KDV iade süreci" → ["KDV iadesi", "iade", "geri ödeme"]
- "akaryakıt istasyonu vergi mükellefiyeti" → ["akaryakıt istasyonu", "vergi mükellefiyeti", "ÖKC"]
- "kredi kartı komisyon stopaj" → ["kredi kartı", "komisyon", "stopaj", "tevkifat"]
- "internet satış kdv" → ["internet satış", "e-ticaret", "KDV", "dijital ticaret"]

SADECE anahtar kelimeleri JSON formatında döndür:
["kelime1", "kelime2", "kelime3", "kelime4"]"""

            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # JSON'u temizle ve parse et
            cleaned = self._clean_json_response(result)
            
            try:
                keywords = json.loads(cleaned)
                if isinstance(keywords, list):
                    # Temizle ve filtrele
                    filtered = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip()) > 1]
                    return json.dumps(filtered[:5], ensure_ascii=False)
            except:
                pass
            
            # JSON parse edilemezse regex ile kelime çıkar
            keywords = self._extract_keywords_from_text(result)
            return json.dumps(keywords[:5], ensure_ascii=False)
                
        except Exception as e:
            print(f"AI anahtar kelime çıkarma başarısız: {str(e)}")
            return self._emergency_fallback(user_query)
    
    def _clean_json_response(self, response: str) -> str:
        """LLM cevabından JSON formatını temizle"""
        response = response.strip()
        
        # Markdown temizleme
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        elif response.startswith('```'):
            response = response.replace('```', '').strip()
        
        # İlk JSON array'i bul
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            return match.group(0)
            
        return response
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Metinden regex ile anahtar kelime çıkar"""
        # Tırnak içindeki kelimeleri bul
        quoted = re.findall(r'"([^"]*)"', text)
        if quoted:
            return quoted[:5]
            
        # Virgül veya satır ile ayrılmış kelimeleri bul
        lines = text.replace(',', '\n').split('\n')
        keywords = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and len(line) > 2:
                keywords.append(line)
                
        return keywords[:5]
    
    def _emergency_fallback(self, user_query: str) -> str:
        """Acil durum fallback - basit kelime çıkarma"""
        try:
            # Büyük harfli kelimeler ve kısaltmalar
            important_words = re.findall(r'\b[A-ZÇĞIÜÖŞ][a-zçğıöşü]*\b', user_query)
            acronyms = re.findall(r'\b[A-ZÇĞIÜÖŞ]{2,}\b', user_query)
            
            # Sayıları ve yılları ekle
            numbers = re.findall(r'\b20\d{2}\b', user_query)
            
            keywords = list(set(important_words + acronyms + numbers))
            
            # Stop word'leri temizle
            stop_words = {'ve', 'ile', 'için', 'hakkında', 'nasıl', 'nedir', 'olan', 'bu'}
            filtered = [kw for kw in keywords if kw.lower() not in stop_words]
            
            return json.dumps(filtered[:3], ensure_ascii=False)
            
        except Exception:
            return '["mevzuat", "vergi", "kanun"]'


def get_gib_mevzuat_tools(llm=None) -> List[BaseTool]:
    """GIB mevzuat araçlarını döndür"""
    return [
        GIBKeywordExtractorTool(llm=llm),
        GIBMevzuatSearchTool(),
        GIBContentFetchTool()
    ]