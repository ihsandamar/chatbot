"""
GIB Mevzuat API Tools
Gelir İdaresi Başkanlığı mevzuat araştırma araçları
"""

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
                "size": 20,
                "sortFieldName": "priority",
                "sortType": "ASC"
            }
            
            # POST data - sadece title ile arama yap, diğerleri boş bırak
            search_data = {
                "kanunType": 1,
                "title": search_terms,
                "kanunNo": "",
                "description": "",
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
                        for item in content[:5]:  # İlk 5 sonucu al
                            result = {
                                "title": item.get("title", ""),
                                "description": item.get("description", "")[:500] + "..." if len(item.get("description", "")) > 500 else item.get("description", ""),
                                "siteLink": item.get("siteLink", ""),
                                "tarih": item.get("tarih", ""),
                                "kanunTitle": item.get("kanunTitle", ""),
                                "kanunNo": item.get("kanunNo", "")
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


class KeywordExtractorTool(BaseTool):
    """AI-driven kullanıcı isteğinden anahtar kelime çıkarma aracı"""
    
    name: str = "extract_keywords"
    description: str = """
    Kullanıcının sorduğu sorudan LLM kullanarak en önemli 3 anahtar kelimeyi çıkarır.
    Bu kelimeler GIB mevzuat aramasında kullanılacak.
    Örnek: 'KDV istisnası hakkında' -> ['KDV', 'istisna', 'vergi']
    """
    
    llm: Any = Field(default=None, exclude=True)
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'llm', llm)
    
    def _run(self, user_query: str) -> str:
        """AI ile anahtar kelimeleri çıkar"""
        try:
            if self.llm:
                # LLM ile anahtar kelime çıkarma
                prompt = f"""
Türkiye vergi ve mali mevzuatı araştırması için anahtar kelime çıkarma:

Kullanıcı sorusu: "{user_query}"

Görevin: Yukarıdaki sorudan GIB (Gelir İdaresi Başkanlığı) mevzuat araması için en uygun 3 anahtar kelimeyi çıkar.

Önemli kurallar:
1. Sadece 3 kelime döndür
2. Vergi terminolojisini öncelikle (KDV, ÖTV, gelir vergisi, kurumlar vergisi, etc.)
3. Türkçe karakter kullanım (ğ, ş, ı, ö, ü, ç)
4. Teknik terimler varsa onları tercihet (tevkifat, tarhiyat, tahakkuk, etc.)
5. Stop word'leri kullanma (ve, ile, için, hakkında, etc.)
6. Sadece arama için kritik kelimeleri seç

Örnekler:
- "KDV iadesi nasıl alınır?" → ["KDV", "iade", "alım"]
- "Özel tüketim vergisi oranları nedir?" → ["özel", "tüketim", "vergisi"]
- "İhracat teşvik belgeleri" → ["ihracat", "teşvik", "belge"]
- "Stopaj oranları 2024" → ["stopaj", "oran", "2024"]

SADECE 3 kelimeyi JSON array formatında döndür:
["kelime1", "kelime2", "kelime3"]
"""
                
                try:
                    response = self.llm.invoke(prompt)
                    result = response.content if hasattr(response, 'content') else str(response)
                    
                    # JSON formatını temizle
                    result = result.strip()
                    if result.startswith('```json'):
                        result = result.replace('```json', '').replace('```', '').strip()
                    elif result.startswith('```'):
                        result = result.replace('```', '').strip()
                    
                    # JSON parse et
                    try:
                        keywords = json.loads(result)
                        if isinstance(keywords, list) and len(keywords) <= 3:
                            return json.dumps(keywords[:3], ensure_ascii=False)
                    except:
                        # JSON parse edilemezse, parantez içindeki kelimeleri al
                        import re
                        matches = re.findall(r'"([^"]*)"', result)
                        if matches:
                            return json.dumps(matches[:3], ensure_ascii=False)
                    
                except Exception as e:
                    print(f"LLM anahtar kelime çıkarma hatası: {e}")
                    # Fallback olarak rule-based yaklaşım
                    pass
            
            # Fallback: Rule-based yaklaşım
            return self._fallback_keyword_extraction(user_query)
            
        except Exception as e:
            return f"Anahtar kelime çıkarma hatası: {str(e)}"
    
    def _fallback_keyword_extraction(self, user_query: str) -> str:
        """Fallback rule-based anahtar kelime çıkarma"""
        try:
            # Türkçe stop words
            stop_words = {
                'bir', 'bu', 'da', 'de', 'den', 'ile', 'için', 'mi', 'mu', 'mü', 'var', 'yok',
                'olan', 'olan', 'olur', 'ama', 'ancak', 'fakat', 'lakin', 'hakkında', 'ilgili',
                'nasıl', 'neden', 'niçin', 'ne', 'kim', 'hangi', 'kaç', 'çok', 'az', 'fazla',
                've', 'veya', 'ya', 'yahut', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
                'şey', 'zaman', 'yer', 'kişi', 'kez', 'defa', 'sefer', 'tane', 'adet', 'nedir',
                'nasıl', 'aldığım', 'alınır', 'yapılır', 'edilir', 'olarak'
            }
            
            # Önemli vergi terimleri
            important_terms = {
                'kdv', 'katma', 'değer', 'vergisi', 'özel', 'tüketim', 'ötv', 'gelir', 'kurumlar',
                'damga', 'harç', 'banka', 'sigorta', 'muameleleri', 'bsmv', 'istisna', 'iade',
                'indirim', 'tevkifat', 'stopaj', 'beyanname', 'tahakkuk', 'tahsilat', 'tarhiyat',
                'ceza', 'faiz', 'gecikme', 'ödeme', 'mahsup', 'terkin', 'iptal', 'düzeltme',
                'ihracat', 'ithalat', 'gümrük', 'transit', 'antrepo', 'serbest', 'bölge',
                'muafiyet', 'muhtasar', 'geçici', 'vergilendir', 'matrah', 'oran', 'haddi',
                'teşvik', 'destek', 'belge', 'sertifika', 'başvuru', 'değerlendirme',
                'ökc', 'okc', 'kamulaştırma', 'kamulaştırım', 'benzinlik', 'akaryakıt', 'station',
                'petrol', 'motorin', 'lpg', 'yakıt', 'istasyon', 'bayii', 'distribütör',
                'lisans', 'ruhsat', 'izin', 'yetki', 'çevre', 'sağlık', 'güvenlik'
            }
            
            # Metni temizle ve küçük harfe çevir
            text = user_query.lower()
            # Türkçe karakterleri koruyarak temizle
            text = re.sub(r'[^\w\s]', ' ', text)
            # Çoklu boşlukları tek boşluğa çevir
            text = re.sub(r'\s+', ' ', text)
            
            # Kelimeleri ayır
            words = text.split()
            
            # Anahtar kelimeleri bul
            keywords = []
            
            # Önce önemli terimleri ara
            for word in words:
                if word in important_terms and word not in keywords:
                    keywords.append(word)
                    if len(keywords) >= 3:
                        break
            
            # Eksik varsa diğer kelimelerden ekle
            if len(keywords) < 3:
                for word in words:
                    if (len(word) > 2 and 
                        word not in stop_words and 
                        word not in keywords and
                        not word.isdigit()):
                        keywords.append(word)
                        if len(keywords) >= 3:
                            break
            
            # En az 1 kelime olsun - hiç kelime yoksa force olarak al
            if not keywords:
                # Tüm kelimeleri dene (stop words hariç)
                all_words = [word for word in words if len(word) > 1 and word not in stop_words]
                if all_words:
                    keywords = all_words[:3]
                else:
                    # Son çare: tüm kelimeleri al
                    keywords = [word for word in words if len(word) > 1][:3]
            
            # Hiç kelime yoksa orijinal sorgudan ilk kelimeleri al
            if not keywords:
                raw_words = user_query.lower().split()
                keywords = [word for word in raw_words if len(word) > 1][:3]
            
            # Yine de hiç yoksa genel terim kullan
            if not keywords:
                keywords = ["mevzuat", "kanun", "düzenleme"]
            
            return json.dumps(keywords[:3], ensure_ascii=False)
            
        except Exception as e:
            return f"Fallback anahtar kelime çıkarma hatası: {str(e)}"


def get_gib_mevzuat_tools(llm=None) -> List[BaseTool]:
    """GIB mevzuat araçlarını döndür"""
    return [
        KeywordExtractorTool(llm=llm),
        GIBMevzuatSearchTool(),
        GIBContentFetchTool()
    ]