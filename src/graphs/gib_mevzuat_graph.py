"""
GIB Mevzuat Araştırma Graph
Gelir İdaresi Başkanlığı mevzuat veritabanında araştırma yapan graph
"""

import json
from typing import List, Dict, Any
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM, State
from src.tools.gib_mevzuat_tools import get_gib_mevzuat_tools


@register_graph("gib_mevzuat")
class GIBMevzuatGraph(BaseGraph):
    """GIB Mevzuat araştırma graph'ı"""
    
    def __init__(self, llm: LLM):
        super().__init__(llm, State)
        self.tools = get_gib_mevzuat_tools(llm=llm.get_chat())
        self.mevzuat_types = [
            "madde", "gerekce", "bkk", "cbk", "teblig", 
            "yonetmelikler", "genelYazilar", "icGenelge", 
            "ozelge", "sirkuler"
        ]
        
    def extract_keywords_node(self, state: State) -> State:
        """Kullanıcı isteğinden anahtar kelimeleri çıkar"""
        try:
            # Kullanıcı girdisini al
            user_input = self._get_user_input(state)
            
            if not user_input:
                return self._add_error_message(state, "Lütfen aramak istediğiniz konuyu belirtin.")
            
            # Anahtar kelime çıkarma aracını kullan
            keyword_tool = next(tool for tool in self.tools if tool.name == "extract_keywords")
            keywords_json = keyword_tool._run(user_input)
            
            try:
                keywords = json.loads(keywords_json)
                if not isinstance(keywords, list) or not keywords:
                    # Fallback: basit kelime ayırma
                    keywords = self._simple_keyword_extraction(user_input)
            except:
                keywords = self._simple_keyword_extraction(user_input)
            
            # State'e kaydet
            state["search_keywords"] = keywords
            state["original_query"] = user_input
            
            print(f"Çıkarılan anahtar kelimeler: {keywords}")
            
            return state
            
        except Exception as e:
            return self._add_error_message(state, f"Anahtar kelime çıkarma hatası: {str(e)}")
    
    def search_mevzuat_node(self, state: State) -> State:
        """Mevzuat arama yap - önce önemli türlerde ara"""
        try:
            keywords = state.get("search_keywords", [])
            if not keywords:
                return self._add_error_message(state, "Arama için anahtar kelime bulunamadı.")
            
            # Arama terimi oluştur
            search_term = " ".join(keywords[:3]).strip()
            if not search_term:
                search_term = state.get("original_query", "mevzuat")
            
            print(f"Arama terimi: '{search_term}'")
            
            # Arama aracını al
            search_tool = next(tool for tool in self.tools if tool.name == "gib_mevzuat_search")
            
            all_results = []
            
            # Öncelikle önemli mevzuat türlerinde ara
            priority_types = ["genelYazilar", "teblig", "yonetmelikler", "icGenelge"]
            
            for mevzuat_type in priority_types:
                try:
                    print(f"Araniyor: {mevzuat_type} - {search_term}")
                    
                    result = search_tool._run(
                        search_terms=search_term,
                        mevzuat_type=mevzuat_type
                    )
                    
                    # JSON formatında sonuç var mı kontrol et
                    if result and result.startswith('[') and result.endswith(']'):
                        try:
                            parsed_results = json.loads(result)
                            if isinstance(parsed_results, list):
                                for item in parsed_results:
                                    item["mevzuat_type"] = mevzuat_type
                                    all_results.append(item)
                                    
                                print(f"  -> {len(parsed_results)} sonuç bulundu")
                        except json.JSONDecodeError:
                            print(f"  -> JSON parse hatası: {result[:100]}")
                    else:
                        print(f"  -> Sonuç yok: {result[:100] if result else 'Boş'}")
                    
                    # Yeterli sonuç bulduysak diğer türleri arama
                    if len(all_results) >= 10:
                        break
                        
                except Exception as e:
                    print(f"  -> {mevzuat_type} arama hatası: {str(e)}")
                    continue
            
            # Sonuçları state'e kaydet
            state["search_results"] = all_results
            
            print(f"Toplam {len(all_results)} sonuç bulundu")
            
            return state
            
        except Exception as e:
            return self._add_error_message(state, f"Mevzuat arama hatası: {str(e)}")
    
    def analyze_results_node(self, state: State) -> State:
        """Sonuçları analiz et ve en uygun olanları seç"""
        try:
            results = state.get("search_results", [])
            keywords = state.get("search_keywords", [])
            
            if not results:
                return self._add_error_message(state, f"'{' '.join(keywords) if keywords else 'Aramanız'}' konusunda mevzuat bulunamadı.")
            
            print(f"Analiz ediliyor: {len(results)} sonuç")
            
            # Tüm sonuçları puanla
            for result in results:
                score = 0
                title = result.get("title", "").lower()
                description = result.get("description", "").lower()
                
                # Anahtar kelime puanlaması
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    # Başlıkta tam eşleşme
                    if keyword_lower in title:
                        score += 5
                    # Açıklamada eşleşme
                    if keyword_lower in description:
                        score += 2
                    # Kısmi eşleşme (3+ karakter)
                    if len(keyword_lower) >= 3:
                        if any(keyword_lower in word for word in title.split()):
                            score += 1
                
                # Tarih yeniliği bonusu
                tarih = result.get("tarih", "")
                if "2024" in tarih or "2023" in tarih:
                    score += 1
                
                result["relevance_score"] = score
            
            # Puanına göre sırala ve en iyileri al
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # En az 1 puanı olan sonuçları al (maksimum 5)
            top_results = [r for r in results if r.get("relevance_score", 0) > 0][:5]
            
            # Hiç puanı olanı yoksa ilk 3'ü al
            if not top_results:
                top_results = results[:3]
            
            state["top_results"] = top_results
            
            print(f"En alakalı {len(top_results)} sonuç seçildi")
            for result in top_results:
                print(f"  - {result.get('title', 'Başlık yok')[:50]}... (Puan: {result.get('relevance_score', 0)})")
            
            return state
            
        except Exception as e:
            return self._add_error_message(state, f"Sonuç analizi hatası: {str(e)}")
    
    def fetch_content_node(self, state: State) -> State:
        """Seçilen sonuçların tam içeriğini al (opsiyonel)"""
        try:
            top_results = state.get("top_results", [])
            
            if not top_results:
                return state
            
            print(f"İçerik detayları alınıyor: {len(top_results)} sonuç")
            
            # İçerik alma aracını al
            content_tool = next((tool for tool in self.tools if tool.name == "gib_content_fetch"), None)
            
            detailed_results = []
            
            for i, result in enumerate(top_results, 1):
                site_link = result.get("siteLink", "")
                
                # İçerik alma işlemini sadece ilk 3 sonuç için yap (performans)
                if i <= 3 and site_link and content_tool:
                    try:
                        print(f"  {i}. İçerik alınıyor: {site_link[:50]}...")
                        
                        content = content_tool._run(site_link)
                        if content and len(content.strip()) > 50:
                            result["full_content"] = content
                            print(f"     -> Başarılı ({len(content)} karakter)")
                        else:
                            result["full_content"] = result.get("description", "İçerik alınamadı")
                            print(f"     -> Kısa içerik, açıklama kullanıldı")
                            
                    except Exception as e:
                        print(f"     -> İçerik alma hatası: {str(e)}")
                        result["full_content"] = result.get("description", "İçerik alınamadı")
                else:
                    # İçerik alınmayan sonuçlar için açıklamayı kullan
                    result["full_content"] = result.get("description", "Açıklama mevcut değil")
                
                detailed_results.append(result)
            
            state["detailed_results"] = detailed_results
            
            return state
            
        except Exception as e:
            print(f"İçerik alma genel hatası: {str(e)}")
            # Hata durumunda top_results'ı detailed_results olarak kullan
            state["detailed_results"] = state.get("top_results", [])
            return state
    
    def generate_summary_node(self, state: State) -> State:
        """Bulunan mevzuatları özetle ve kullanıcıya sun"""
        try:
            detailed_results = state.get("detailed_results", [])
            top_results = state.get("top_results", [])
            original_query = state.get("original_query", "")
            
            # Sonuç yoksa hata mesajı
            if not detailed_results and not top_results:
                return self._add_bot_message(state, "Aramanızla ilgili mevzuat bulunamadı.")
            
            # Mevcut sonuçları kullan
            results_to_use = detailed_results if detailed_results else top_results
            
            # Basit özet oluştur
            summary = f"🎯 KONU: {original_query}\n\n"
            summary += f"📋 BULUNAN MEVZUAT ({len(results_to_use)} adet):\n\n"
            
            for i, result in enumerate(results_to_use, 1):
                title = result.get('title', 'Başlık yok')
                tarih = result.get('tarih', 'Tarih yok')
                kanun_title = result.get('kanunTitle', '')
                kanun_no = result.get('kanunNo', '')
                mevzuat_type = result.get('mevzuat_type', 'Genel')
                site_link = result.get('siteLink', '')
                description = result.get('description', '')[:200]
                
                summary += f"{i}. **{title}**\n"
                summary += f"   📅 Tarih: {tarih}\n"
                if kanun_title:
                    summary += f"   📜 Kanun: {kanun_title}"
                    if kanun_no:
                        summary += f" ({kanun_no})"
                    summary += "\n"
                summary += f"   📂 Tür: {mevzuat_type}\n"
                if description:
                    summary += f"   📝 Açıklama: {description}...\n"
                if site_link:
                    summary += f"   🔗 Link: {site_link}\n"
                summary += "\n"
            
            # Özeti state'e kaydet ve bot mesajı olarak ekle
            state["mevzuat_summary"] = summary
            return self._add_bot_message(state, summary)
            
        except Exception as e:
            return self._add_error_message(state, f"Özet oluşturma hatası: {str(e)}")
    
    def _get_user_input(self, state: State) -> str:
        """State'den kullanıcı girdisini al"""
        # Önce user_query'yi kontrol et
        user_input = state.get("user_query", "")
        
        # Yoksa messages'dan son human mesajını al
        if not user_input:
            messages = state.get("messages", [])
            for message in reversed(messages):
                # Message objectı mı kontrol et
                if hasattr(message, 'type') and message.type == 'human':
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            # Content list formatı
                            for content_part in message.content:
                                if isinstance(content_part, dict) and content_part.get('type') == 'text':
                                    user_input = content_part.get('text', '')
                                    if user_input.strip():
                                        break
                        elif isinstance(message.content, str):
                            user_input = message.content
                        break
                # String mesaj formatı (eski format)
                elif isinstance(message, str) and not message.startswith('Bot:'):
                    user_input = message
                    break
        
        return user_input.strip() if user_input else ""
    
    def _add_bot_message(self, state: State, message: str) -> State:
        """Bot mesajını doğru formatta ekle"""
        from langchain.schema import AIMessage
        
        messages = state.get("messages", [])
        bot_message = AIMessage(content=message)
        messages.append(bot_message)
        state["messages"] = messages
        return state
    
    def _add_error_message(self, state: State, error: str) -> State:
        """Hata mesajını ekle"""
        return self._add_bot_message(state, f"❌ Hata: {error}")
    
    def _simple_keyword_extraction(self, text: str) -> list:
        """Basit anahtar kelime çıkarma (fallback)"""
        import re
        
        # Önemli vergi terimleri
        important_terms = [
            'kdv', 'katma', 'değer', 'vergisi', 'özel', 'tüketim', 'ötv', 'gelir', 'kurumlar',
            'tevkifat', 'stopaj', 'iade', 'istisna', 'muafiyet', 'indirim', 'beyanname',
            'ihracat', 'ithalat', 'fatura', 'belge', 'teşvik', 'matrah', 'oran', 'tarhiyat'
        ]
        
        # Metni temizle
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Önemli terimleri bul
        keywords = []
        for word in words:
            if word in important_terms and word not in keywords:
                keywords.append(word)
                if len(keywords) >= 3:
                    break
        
        # Yeterli değilse diğer kelimeleri ekle
        if len(keywords) < 3:
            stop_words = {'bir', 'bu', 'da', 'de', 'den', 'ile', 'için', 'mi', 'mu', 'nı', 'nü', 'na', 'ne', 'hakkında', 'nasıl', 'nedir'}
            for word in words:
                if (len(word) > 2 and 
                    word not in stop_words and 
                    word not in keywords and
                    not word.isdigit()):
                    keywords.append(word)
                    if len(keywords) >= 3:
                        break
        
        return keywords[:3] if keywords else ['mevzuat', 'kanun', 'hüküm']
    
    def build_graph(self):
        """Graph'ı oluştur"""
        print("GIB Mevzuat araştırma graph'ı çalıştırılıyor...")
        
        graph = StateGraph(State)
        
        # Node'ları ekle
        graph.add_node("extract_keywords", RunnableLambda(self.extract_keywords_node))
        graph.add_node("search_mevzuat", RunnableLambda(self.search_mevzuat_node))
        graph.add_node("analyze_results", RunnableLambda(self.analyze_results_node))
        graph.add_node("fetch_content", RunnableLambda(self.fetch_content_node))
        graph.add_node("generate_summary", RunnableLambda(self.generate_summary_node))
        
        # Edge'leri tanımla
        graph.set_entry_point("extract_keywords")
        graph.add_edge("extract_keywords", "search_mevzuat")
        graph.add_edge("search_mevzuat", "analyze_results")
        graph.add_edge("analyze_results", "fetch_content")
        graph.add_edge("fetch_content", "generate_summary")
        graph.set_finish_point("generate_summary")
        
        return graph.compile()