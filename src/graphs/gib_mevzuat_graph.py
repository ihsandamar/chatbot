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
                state["messages"] = state.get("messages", []) + [
                    "Bot: Lütfen aramak istediğiniz konuyu belirtin."
                ]
                return state
            
            # Anahtar kelime çıkarma aracını kullan
            keyword_tool = next(tool for tool in self.tools if tool.name == "extract_keywords")
            keywords_json = keyword_tool._run(user_input)
            keywords = json.loads(keywords_json)
            
            # State'e kaydet
            state["search_keywords"] = keywords
            state["original_query"] = user_input
            
            print(f"Çıkarılan anahtar kelimeler: {keywords}")
            
            return state
            
        except Exception as e:
            state["messages"] = state.get("messages", []) + [
                f"Bot: Anahtar kelime çıkarma hatası: {str(e)}"
            ]
            return state
    
    def search_mevzuat_node(self, state: State) -> State:
        """Tüm mevzuat türlerinde arama yap"""
        try:
            keywords = state.get("search_keywords", [])
            if not keywords or len(keywords) == 0:
                # Fallback: kullanıcı girdisini direkt kullan
                user_input = self._get_user_input(state)
                if user_input:
                    keywords = ["mevzuat", user_input.split()[0] if user_input.split() else "kanun"]
                    state["search_keywords"] = keywords
                else:
                    state["messages"] = state.get("messages", []) + [
                        "Bot: Arama için anahtar kelime bulunamadı."
                    ]
                    return state
            
            # Arama terimi oluştur (ilk 3 kelimeyi birleştir)
            search_term = " ".join(keywords[:3]).strip()
            
            # Boş arama terimi kontrolü
            if not search_term or search_term == "":
                user_input = self._get_user_input(state)
                search_term = user_input if user_input else "mevzuat"
            
            # Arama aracını al
            search_tool = next(tool for tool in self.tools if tool.name == "gib_mevzuat_search")
            
            all_results = []
            
            # Her mevzuat türünde ara
            for mevzuat_type in self.mevzuat_types:
                try:
                    print(f"{mevzuat_type} türünde aranıyor: {search_term}")
                    
                    result = search_tool._run(
                        search_terms=search_term,
                        mevzuat_type=mevzuat_type
                    )
                    
                    # JSON formatında sonuç dönerse parse et
                    if result.startswith('[') and result.endswith(']'):
                        parsed_results = json.loads(result)
                        for item in parsed_results:
                            item["mevzuat_type"] = mevzuat_type
                            all_results.append(item)
                    
                except Exception as e:
                    print(f"{mevzuat_type} arama hatası: {str(e)}")
                    continue
            
            # Sonuçları state'e kaydet
            state["search_results"] = all_results
            
            if all_results:
                print(f"Toplam {len(all_results)} sonuç bulundu")
            else:
                print("Hiç sonuç bulunamadı")
            
            return state
            
        except Exception as e:
            state["messages"] = state.get("messages", []) + [
                f"Bot: Mevzuat arama hatası: {str(e)}"
            ]
            return state
    
    def analyze_results_node(self, state: State) -> State:
        """Sonuçları analiz et ve en uygun olanları seç"""
        try:
            results = state.get("search_results", [])
            original_query = state.get("original_query", "")
            keywords = state.get("search_keywords", [])
            
            if not results:
                state["messages"] = state.get("messages", []) + [
                    f"Bot: '{' '.join(keywords)}' konusunda herhangi bir mevzuat bulunamadı."
                ]
                return state
            
            # En alakalı sonuçları seç (başlık ve açıklamada anahtar kelimeleri ara)
            relevant_results = []
            
            for result in results:
                score = 0
                title = result.get("title", "").lower()
                description = result.get("description", "").lower()
                
                # Anahtar kelime puanlaması
                for keyword in keywords:
                    if keyword.lower() in title:
                        score += 3  # Başlıkta geçerse yüksek puan
                    if keyword.lower() in description:
                        score += 1  # Açıklamada geçerse düşük puan
                
                if score > 0:
                    result["relevance_score"] = score
                    relevant_results.append(result)
            
            # Puanına göre sırala
            relevant_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # En iyi 3'ünü al
            top_results = relevant_results[:3]
            
            state["top_results"] = top_results
            
            if top_results:
                print(f"En alakalı {len(top_results)} sonuç seçildi")
            
            return state
            
        except Exception as e:
            state["messages"] = state.get("messages", []) + [
                f"Bot: Sonuç analizi hatası: {str(e)}"
            ]
            return state
    
    def fetch_content_node(self, state: State) -> State:
        """Seçilen sonuçların tam içeriğini al"""
        try:
            top_results = state.get("top_results", [])
            
            if not top_results:
                return state
            
            # İçerik alma aracını al
            content_tool = next(tool for tool in self.tools if tool.name == "gib_content_fetch")
            
            detailed_results = []
            
            for result in top_results:
                site_link = result.get("siteLink", "")
                if site_link:
                    try:
                        print(f"İçerik alınıyor: {site_link}")
                        
                        content = content_tool._run(site_link)
                        result["full_content"] = content
                        detailed_results.append(result)
                        
                    except Exception as e:
                        print(f"İçerik alma hatası: {str(e)}")
                        result["full_content"] = result.get("description", "İçerik alınamadı")
                        detailed_results.append(result)
                else:
                    result["full_content"] = result.get("description", "Link bulunamadı")
                    detailed_results.append(result)
            
            state["detailed_results"] = detailed_results
            
            return state
            
        except Exception as e:
            state["messages"] = state.get("messages", []) + [
                f"Bot: İçerik alma hatası: {str(e)}"
            ]
            return state
    
    def generate_summary_node(self, state: State) -> State:
        """Bulunan mevzuatları özetle ve kullanıcıya sun"""
        try:
            detailed_results = state.get("detailed_results", [])
            original_query = state.get("original_query", "")
            
            if not detailed_results:
                state["messages"] = state.get("messages", []) + [
                    "Bot: Aramanızla ilgili mevzuat bulunamadı."
                ]
                return state
            
            # LLM ile özet oluştur
            summary_prompt = f"""
            Kullanıcı şunu sordu: "{original_query}"

            Aşağıdaki GIB mevzuatları bulundu:

            """
            
            for i, result in enumerate(detailed_results, 1):
                summary_prompt += f"""
                {i}. {result.get('title', 'Başlık yok')}
                Tarih: {result.get('tarih', 'Tarih yok')}
                Kanun: {result.get('kanunTitle', '')} ({result.get('kanunNo', '')})
                Tür: {result.get('mevzuat_type', 'Bilinmiyor')}
                
                İçerik:
                {result.get('full_content', 'İçerik yok')[:1000]}...
                
                ---
                """
            
            summary_prompt += """
            Yukarıdaki mevzuatları kullanıcının sorusu bağlamında özetle:
            1. Kullanıcının sorusuna doğrudan cevap ver
            2. İlgili mevzuatları kısaca açıkla
            3. Önemli hükümler ve kurallar belirt
            4. Pratik uygulamaya yönelik bilgi ver
            5. Türkçe, anlaşılır ve yapılandırılmış bir yanıt hazırla
            
            Yanıt formatı:
            🎯 KONU: [Kullanıcının sorusu]
            
            📋 BULUNAN MEVZUAT:
            - [Mevzuat listesi]
            
            ⚖️ ÖZET VE AÇIKLAMALAR:
            [Detaylı açıklamalar]
            
            💡 ÖNEMLİ NOKTALAR:
            [Önemli hususlar]
            """
            
            try:
                response = self.llm.get_chat().invoke(summary_prompt)
                summary = response.content if hasattr(response, 'content') else str(response)
                
                state["messages"] = state.get("messages", []) + [f"Bot: {summary}"]
                
            except Exception as e:
                # Fallback özet
                fallback_summary = f"🎯 KONU: {original_query}\n\n📋 BULUNAN MEVZUAT:\n"
                
                for i, result in enumerate(detailed_results, 1):
                    fallback_summary += f"{i}. {result.get('title', 'Başlık yok')}\n"
                    fallback_summary += f"   📅 Tarih: {result.get('tarih', 'Bilinmiyor')}\n"
                    fallback_summary += f"   📜 Kanun: {result.get('kanunTitle', 'Bilinmiyor')}\n"
                    fallback_summary += f"   🔗 Link: {result.get('siteLink', 'Yok')}\n\n"
                
                fallback_summary += "⚠️ Detaylı özet oluşturulamadı. Yukarıdaki linkleri inceleyebilirsiniz."
                
                state["messages"] = state.get("messages", []) + [f"Bot: {fallback_summary}"]
            
            return state
            
        except Exception as e:
            state["messages"] = state.get("messages", []) + [
                f"Bot: Özet oluşturma hatası: {str(e)}"
            ]
            return state
    
    def _get_user_input(self, state: State) -> str:
        """State'den kullanıcı girdisini al"""
        # user_query varsa onu kullan
        user_input = state.get("user_query", "")
        
        # Yoksa messages'dan al
        if not user_input:
            messages = state.get("messages", [])
            for message in reversed(messages):
                if hasattr(message, 'type') and message.type == 'human':
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            for content_part in message.content:
                                if isinstance(content_part, dict) and content_part.get('type') == 'text':
                                    user_input = content_part.get('text', '')
                                    break
                        else:
                            user_input = message.content
                        break
        
        return user_input.strip()
    
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