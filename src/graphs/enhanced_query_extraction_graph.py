# src/graphs/enhanced_query_extraction_graph.py
"""
Enhanced Query Extraction Graph - SQL tablo şemasını çeker ve gelişmiş intent analizi yapar
"""

import time
import re
import json
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AnyMessage
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph.message import add_messages

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.services.app_logger import log
from src.graphs.generic_sql_graph import SQLAgentConfiguration


# State Schema
class EnhancedText2SQLState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    table_info: Optional[Dict[str, Any]]
    extraction_result: Optional[str]
    errors: Optional[Dict[str, Any]]
    
    # Enhanced intent analysis state
    intent_analysis: Optional[Dict[str, Any]]
    analysis_result: Optional[str]
    context_analysis: Optional[Dict[str, Any]]


class Table(BaseModel):
    """Table in SQL database with detailed schema information."""
    name: str = Field(description="Name of table in SQL database.")
    columns: List[Dict[str, Any]] = Field(default=[], description="Column definitions with distinct values")
    relations: List[str] = Field(default=[], description="Table relations")


@register_graph("enhanced_query_extraction")
class EnhancedQueryExtractionGraph(BaseGraph):
    """Enhanced Query Extraction Graph - gelişmiş context-aware intent analizi"""
    
    def __init__(self, llm, db: SQLDatabase = None, memory: MemorySaver = None):
        super().__init__(llm=llm, state_class=EnhancedText2SQLState, memory=memory)
        self.logger = log.get(module="enhanced_query_extraction_graph")
        self.db = db
        self.prepare_table_data_tool = self._create_prepare_table_data_tool()
        self.enhanced_analyze_intent_tool = self._create_enhanced_analyze_intent_tool()
        
        self.logger.info("Enhanced Query Extraction Graph initialized")
    
    def _is_string_column(self, column_type: str) -> bool:
        """Kolonun string tipinde olup olmadığını kontrol et"""
        string_types = ['varchar', 'text', 'char', 'string', 'nvarchar', 'ntext']
        return any(str_type in column_type.lower() for str_type in string_types)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation - 4 characters ≈ 1 token"""
        return max(1, len(text) // 4)

    def _extract_key_entities(self, messages: List[AnyMessage]) -> List[str]:
        """Extract key entities from messages"""
        entities = set()
        for msg in messages:
            content = msg.content.upper()
            # Look for common business entities (uppercase words, numbers)
            words = content.split()
            for word in words:
                if len(word) > 2 and (word.isupper() or word.isdigit()):
                    entities.add(word)
        return list(entities)[:5]  # Limit to 5 entities

    def _extract_topics(self, messages: List[AnyMessage]) -> List[str]:
        """Extract topics from messages"""
        topics = set()
        business_keywords = ['satış', 'müşteri', 'ürün', 'fatura', 'sipariş', 'stok', 'personel', 'departman']
        for msg in messages:
            content = msg.content.lower()
            for keyword in business_keywords:
                if keyword in content:
                    topics.add(keyword)
        return list(topics)[:3]  # Limit to 3 topics

    def _create_conversation_summary(self, messages: List[AnyMessage], token_budget: int) -> str:
        """Eski mesajları özetle"""
        if not messages:
            return ""
        
        # Key entities ve topics çıkar
        entities = self._extract_key_entities(messages)
        topics = self._extract_topics(messages)
        
        summary = f"Önceki konuşma: {', '.join(topics)} konularında {', '.join(entities)} hakkında sorular."
        
        # Token budget içinde tut
        if self._estimate_tokens(summary) > token_budget:
            summary = summary[:token_budget*4]  # Rough cut
        
        return summary

    def _detect_conversation_flow(self, messages: List[AnyMessage]) -> Dict:
        """Konuşma akış göstergelerini tespit et"""
        flow_indicators = {
            "pronoun_references": [],
            "flow_words": [],
            "topic_continuity": False,
            "entity_continuity": []
        }
        
        if not messages:
            return flow_indicators
        
        current_content = messages[-1].content.lower()
        
        # Pronoun references
        pronouns = ["bunların", "onların", "şunların", "olanın", "olan", "bunun", "onun"]
        flow_indicators["pronoun_references"] = [p for p in pronouns if p in current_content]
        
        # Flow words
        flow_words = ["peki", "şimdi", "sonra", "ayrıca", "de", "da", "bir de"]
        flow_indicators["flow_words"] = [w for w in flow_words if w in current_content]
        
        # Entity continuity check
        if len(messages) > 1:
            prev_content = messages[-2].content.lower() if len(messages) > 1 else ""
            common_words = set(current_content.split()) & set(prev_content.split())
            business_words = [w for w in common_words if len(w) > 3 and w not in ["olan", "için", "nedir"]]
            flow_indicators["entity_continuity"] = list(business_words)
            flow_indicators["topic_continuity"] = len(business_words) > 0
        
        return flow_indicators

    def _get_smart_context(self, messages: List[AnyMessage], max_tokens: int = 1500) -> Dict:
        """Akıllı context selection - token budget dahilinde"""
        
        context_result = {
            "current_message": "",
            "recent_context": [],
            "conversation_summary": "",
            "flow_indicators": {},
            "total_tokens": 0
        }
        
        if not messages:
            return context_result
        
        # 1. Current message (always include)
        current_msg = messages[-1]
        context_result["current_message"] = current_msg.content
        current_tokens = self._estimate_tokens(current_msg.content)
        context_result["total_tokens"] += current_tokens
        
        # 2. Recent context (backwards scan)
        recent_messages = []
        token_budget = max_tokens * 0.7  # %70 budget for recent context
        
        for msg in reversed(messages[:-1]):
            msg_tokens = self._estimate_tokens(msg.content)
            if context_result["total_tokens"] + msg_tokens > token_budget:
                break
            recent_messages.insert(0, msg)  # Maintain chronological order
            context_result["total_tokens"] += msg_tokens
            
            # Stop at natural break points
            if len(recent_messages) >= 8:  # Max 8 recent messages
                break
        
        context_result["recent_context"] = recent_messages
        
        # 3. Conversation summary for older messages
        remaining_budget = max_tokens - context_result["total_tokens"]
        older_messages = messages[:-len(recent_messages)-1]
        
        if older_messages and remaining_budget > 300:  # Min 300 tokens for summary
            summary = self._create_conversation_summary(older_messages, remaining_budget)
            context_result["conversation_summary"] = summary
            context_result["total_tokens"] += self._estimate_tokens(summary)
        
        # 4. Flow indicators
        context_result["flow_indicators"] = self._detect_conversation_flow(
            recent_messages + [current_msg]
        )
        
        return context_result

    def _create_enhanced_prompt(self, context_data: Dict, table_info: Dict) -> str:
        """Context-aware, structured prompt oluştur"""
        
        current_msg = context_data["current_message"]
        recent_context = context_data["recent_context"] 
        conversation_summary = context_data["conversation_summary"]
        flow_indicators = context_data["flow_indicators"]
        
        # Table context
        table_context = ""
        if table_info and table_info.get("columns"):
            searchable_columns = [
                col["name"] for col in table_info["columns"] 
                if col.get("distinct_values")
            ]
            table_context = f"Aranabilir kolonlar: {', '.join(searchable_columns[:10])}"  # İlk 10
        
        # Recent conversation context
        recent_context_text = ""
        if recent_context:
            for i, msg in enumerate(recent_context):
                msg_type = "Kullanıcı" if hasattr(msg, 'type') and msg.type == "human" else "Asistan"
                recent_context_text += f"{i+1}. {msg_type}: {msg.content[:100]}...\n"
        
        # Flow analysis
        flow_analysis = ""
        if flow_indicators["pronoun_references"]:
            flow_analysis += f"Pronoun referansları: {flow_indicators['pronoun_references']}\n"
        if flow_indicators["flow_words"]:
            flow_analysis += f"Akış kelimeleri: {flow_indicators['flow_words']}\n"
        if flow_indicators["topic_continuity"]:
            flow_analysis += f"Konu devamı: {flow_indicators['entity_continuity']}\n"
        
        prompt = f"""ERP CHATBOT - CONTEXT-AWARE INTENT ANALYSIS

==== CONVERSATION CONTEXT ====
{conversation_summary}

Son Konuşma:
{recent_context_text}

Mevcut Kullanıcı Mesajı: "{current_msg}"

==== CONTEXT ANALYSIS ====
{flow_analysis}

==== TABLE CONTEXT ====  
{table_context}

==== INTENT CLASSIFICATION RULES ====
Intent Türleri:
• count: Sayım ("kaç tane", "sayısı", "adet")
• sum: Toplama ("toplam", "total tutar", "toplamı")  
• list: Listeleme ("göster", "hangileri", "listele")
• group: Gruplama ("kategoriye göre", "dağılım", "gruplayarak")
• filter: Filtreleme ("şartını sağlayan", "kriterlere uygun")
• compare: Karşılaştırma ("farkı", "hangisi daha", "karşılaştır")
• trend: Trend analizi ("seyre git", "artış", "azalış")

Konuşma Türleri:
• new_query: Tamamen yeni konu
• follow_up: Önceki soruyla ilgili ("bunların toplamı", "peki listeler misin")
• drill_down: Önceki sonuçlarda detaya inme ("DOĞALGAZ olanların detayı")
• refinement: Kapsam değişikliği ("bu ay" → "son 3 ay")
• clarification: Belirsizlik giderme ("hangi ABC firması")
• comparison_extension: Karşılaştırma ekleme

==== CONTEXT CLUES DECISION MATRIX ====
"bunların", "onların" → follow_up/drill_down
"peki", "şimdi" → follow_up  
"olan", "olanın" + entity → drill_down
Aynı entity geçiyor → follow_up/refinement
Farklı entity → new_query

==== REQUIRED OUTPUT FORMAT ====
{{
    "intent_type": "count|sum|list|group|filter|compare|trend",
    "conversation_type": "new_query|follow_up|drill_down|refinement|clarification|comparison_extension", 
    "confidence": 0.0-1.0,
    "target_action": "ne yapmaya çalışıyor - kısa açıklama",
    "reasoning": "karar verme süreci açıklaması",
    "requires_context": true/false,
    "context_references": ["hangi kelimelere referans veriyor"],
    "suggested_entities": ["potansiyel entity'ler tabloda aranacak"]
}}

ANALYSIS:"""
        
        return prompt

    def _parse_and_validate_response(self, llm_response: str) -> Dict:
        """LLM yanıtını parse et ve validate et"""
        try:
            # JSON extract (sometimes LLM adds extra text)
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                raise ValueError("JSON bulunamadı")
            
            # Validation
            required_fields = ["intent_type", "conversation_type", "confidence"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = self._get_default_value(field)
            
            # Normalize confidence
            confidence = float(analysis.get("confidence", 0.5))
            analysis["confidence"] = max(0.0, min(1.0, confidence))
            
            return analysis
            
        except Exception as e:
            # Fallback default analysis
            return {
                "intent_type": "list",
                "conversation_type": "new_query",
                "confidence": 0.3,
                "target_action": "Parsing hatası nedeniyle varsayılan analiz",
                "reasoning": f"LLM response parse edilemedi: {str(e)}",
                "requires_context": False,
                "context_references": [],
                "suggested_entities": []
            }

    def _get_default_value(self, field: str):
        """Get default value for missing field"""
        defaults = {
            "intent_type": "list",
            "conversation_type": "new_query", 
            "confidence": 0.5,
            "target_action": "Varsayılan analiz",
            "reasoning": "Eksik alan için varsayılan değer",
            "requires_context": False,
            "context_references": [],
            "suggested_entities": []
        }
        return defaults.get(field, "")

    def _apply_fallback_logic(self, context_data: Dict, analysis_result: Dict) -> Dict:
        """Düşük confidence durumunda fallback logic"""
        
        current_msg = context_data["current_message"].lower()
        flow_indicators = context_data["flow_indicators"]
        
        # Rule-based fallback
        if any(word in current_msg for word in ["kaç", "sayı", "adet"]):
            analysis_result["intent_type"] = "count"
            analysis_result["confidence"] = min(0.8, analysis_result["confidence"] + 0.3)
        
        elif any(word in current_msg for word in ["toplam", "total"]):
            analysis_result["intent_type"] = "sum" 
            analysis_result["confidence"] = min(0.8, analysis_result["confidence"] + 0.3)
        
        elif any(word in current_msg for word in ["göster", "listele"]):
            analysis_result["intent_type"] = "list"
            analysis_result["confidence"] = min(0.8, analysis_result["confidence"] + 0.3)
        
        # Conversation type fallback
        if flow_indicators["pronoun_references"] or flow_indicators["flow_words"]:
            analysis_result["conversation_type"] = "follow_up"
            analysis_result["confidence"] = min(0.8, analysis_result["confidence"] + 0.2)
        
        analysis_result["reasoning"] += " (Fallback kuralları uygulandı)"
        
        return analysis_result

    def _format_analysis_result(self, analysis_result: Dict, context_data: Dict) -> str:
        """Format analysis result for display"""
        return f"""Enhanced Intent Analizi Tamamlandı:

• **Intent Türü:** {analysis_result.get('intent_type', 'N/A')}
• **Konuşma Türü:** {analysis_result.get('conversation_type', 'N/A')}
• **Güven Skoru:** {analysis_result.get('confidence', 0):.2f}
• **Hedef Aksiyon:** {analysis_result.get('target_action', 'N/A')}
• **Context Gerekli:** {'Evet' if analysis_result.get('requires_context') else 'Hayır'}

**Analiz Gerekçesi:** {analysis_result.get('reasoning', 'N/A')}

**Context Referansları:** {', '.join(analysis_result.get('context_references', []))}
**Önerilen Entity'ler:** {', '.join(analysis_result.get('suggested_entities', []))}

**Token Kullanımı:** {context_data.get('total_tokens', 0)}"""

    def _create_prepare_table_data_tool(self):
        """prepare_table_data tool'unu oluştur"""
        
        # Outer scope'dan self referanslarını alalım
        db = self.db
        logger = self.logger
        is_string_column = self._is_string_column
        
        @tool
        def prepare_table_data(table_name: str, table_columns: List[Dict], max_distinct_values: int = 50) -> Dict:
            """Config'ten aldığı bilgiler ile table_info'yu doldurur"""
            try:
                execute_tool = QuerySQLDatabaseTool(db=db)
                enriched_columns = []
                
                for config_col in table_columns:
                    col_name = config_col.get("name", "") if isinstance(config_col, dict) else str(config_col)
                    col_type = config_col.get("type", "") if isinstance(config_col, dict) else ""
                    col_description = config_col.get("description", "") if isinstance(config_col, dict) else ""
                    
                    column_info = {
                        "name": col_name,
                        "type": col_type,
                        "description": col_description,
                        "distinct_values": []
                    }
                    
                    # String kolonlar için distinct values çek
                    if col_name and is_string_column(col_type):
                        try:
                            # Get distinct values
                            query = f"SELECT DISTINCT TOP {max_distinct_values} [{col_name}] FROM [{table_name}] WHERE [{col_name}] IS NOT NULL"
                            result = execute_tool.invoke(query)
                            
                            # Parse result
                            values = []
                            for line in result.split('\n'):
                                line = line.strip()
                                if line and not line.startswith('-') and not line.startswith('('):
                                    values.append(line)
                            
                            column_info["distinct_values"] = values[:max_distinct_values]
                            logger.info(f"Collected {len(values)} distinct values for {col_name}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to get distinct values for {col_name}", error=str(e))
                            column_info["distinct_values"] = []
                    
                    enriched_columns.append(column_info)
                
                # Table model oluştur
                table_info = Table(
                    name=table_name,
                    columns=enriched_columns,
                    relations=[]
                )
                
                # Table info'yu döndür
                return {
                    "table_name": table_info.name,
                    "columns": table_info.columns,
                    "relations": table_info.relations
                }
                
            except Exception as e:
                logger.error("Table data preparation failed", error=str(e))
                raise Exception(f"Table data preparation failed: {str(e)}")
        
        return prepare_table_data

    def _create_enhanced_analyze_intent_tool(self):
        """Enhanced intent analiz tool'unu oluştur"""
        
        # Outer scope'dan referansları al
        llm = self.llm.get_chat()
        logger = self.logger
        get_smart_context = self._get_smart_context
        create_enhanced_prompt = self._create_enhanced_prompt
        parse_and_validate_response = self._parse_and_validate_response
        apply_fallback_logic = self._apply_fallback_logic
        
        @tool
        def enhanced_analyze_intent_tool(messages: List[AnyMessage], table_info: Dict = None) -> Dict:
            """Enhanced context-aware intent analysis"""
            try:
                start_time = time.time()
                
                # 1. Smart context selection
                context_data = get_smart_context(messages, max_tokens=1500)
                
                # 2. Enhanced prompt creation
                enhanced_prompt = create_enhanced_prompt(context_data, table_info or {})
                
                # 3. LLM call with structured prompt
                response = llm.invoke([{"role": "user", "content": enhanced_prompt}])
                
                # 4. Parse and validate result
                analysis_result = parse_and_validate_response(response.content)
                
                # 5. Fallback handling
                if analysis_result["confidence"] < 0.6:
                    analysis_result = apply_fallback_logic(context_data, analysis_result)
                
                # 6. Add performance metrics
                duration = time.time() - start_time
                analysis_result["performance"] = {
                    "duration": f"{duration:.2f}s",
                    "context_messages": len(messages),
                    "total_tokens": context_data.get("total_tokens", 0)
                }
                
                logger.info("Enhanced intent analysis completed", 
                           duration=f"{duration:.2f}s",
                           confidence=f"{analysis_result['confidence']:.2f}",
                           intent=analysis_result["intent_type"])
                
                return {
                    "analysis": analysis_result,
                    "context_data": context_data
                }
                
            except Exception as e:
                logger.error("Enhanced intent analysis failed", error=str(e))
                return {
                    "analysis": {
                        "intent_type": "list",
                        "conversation_type": "new_query", 
                        "confidence": 0.3,
                        "target_action": f"Enhanced analiz başarısız: {str(e)}",
                        "reasoning": f"Enhanced LLM analizi başarısız: {str(e)}",
                        "requires_context": False,
                        "context_references": [],
                        "suggested_entities": []
                    },
                    "context_data": {"total_tokens": 0}
                }
        
        return enhanced_analyze_intent_tool

    def build_graph(self):
        """Build Enhanced Query Extraction graph"""
        self.logger.info("Building Enhanced Query Extraction graph")
        
        try:
            if self.memory is None:
                self.memory = MemorySaver()
            
            graph_builder = StateGraph(EnhancedText2SQLState, config_schema=SQLAgentConfiguration)
            
            # 1. Prepare Table Data Node
            def prepare_table_data_node(state: EnhancedText2SQLState, *, config) -> EnhancedText2SQLState:
                """Tool'u çağırır ve ToolMessage ekler"""
                try:
                    configurable = config.get("configurable", {})
                    
                    # Config'ten table bilgilerini al
                    table_name = configurable.get("table_name", "")
                    config_columns = configurable.get("table_columns", [])
                    max_distinct_values = configurable.get("max_distinct_values", 50)
                    
                    if not table_name or not config_columns:
                        error_msg = "Table name or columns not found in config"
                        state["errors"] = {"error_message": error_msg}
                        state["messages"] = [ToolMessage(content=error_msg, tool_call_id="prepare_table_data")]
                        return state
                    
                    # Tool'u çağır
                    tool_result = self.prepare_table_data_tool.invoke({
                        "table_name": table_name,
                        "table_columns": config_columns,
                        "max_distinct_values": max_distinct_values
                    })
                    
                    # State'e kaydet
                    state["table_info"] = tool_result
                    
                    # Success message
                    columns = tool_result.get("columns", [])
                    searchable_columns = [col["name"] for col in columns if col.get("distinct_values")]
                    success_msg = f"""Tablo şeması başarıyla çekildi:

• **Tablo:** {table_name}
• **Kolonlar:** {len(columns)} adet
• **Aranabilir Kolonlar:** {len(searchable_columns)} adet

**Distinct değerler çekilen kolonlar:**
{', '.join(searchable_columns)}"""
                    
                    state["extraction_result"] = success_msg
                    state["messages"] = [ToolMessage(content=success_msg, tool_call_id="prepare_table_data")]
                    
                    self.logger.info("Table schema extracted successfully", 
                                   table=table_name, 
                                   columns=len(columns),
                                   searchable=len(searchable_columns))
                    return state
                    
                except Exception as e:
                    error_msg = f"Tablo şeması çekilirken hata oluştu: {str(e)}"
                    state["errors"] = {"error_message": error_msg}
                    state["extraction_result"] = error_msg
                    state["messages"] = [ToolMessage(content=error_msg, tool_call_id="prepare_table_data")]
                    
                    self.logger.error("Table schema extraction failed", error=str(e))
                    return state
            
            # 2. Enhanced Analyze Intent Node
            def enhanced_analyze_intent_node(state: EnhancedText2SQLState, *, config) -> EnhancedText2SQLState:
                """Enhanced intent analysis with smart context"""
                try:
                    messages = state.get("messages", [])
                    table_info = state.get("table_info", {})
                    
                    if not messages:
                        error_msg = "No messages found for intent analysis"
                        state["errors"] = {"error_message": error_msg}
                        state["messages"] = [ToolMessage(content=error_msg, tool_call_id="enhanced_analyze_intent")]
                        return state
                    
                    # Enhanced tool çağrısı
                    tool_result = self.enhanced_analyze_intent_tool.invoke({
                        "messages": messages,
                        "table_info": table_info
                    })
                    
                    # Sonuçları state'e kaydet
                    analysis = tool_result.get("analysis", {})
                    context_data = tool_result.get("context_data", {})
                    
                    state["intent_analysis"] = analysis
                    state["context_analysis"] = context_data
                    
                    # Success message
                    success_msg = self._format_analysis_result(analysis, context_data)
                    state["analysis_result"] = success_msg
                    state["messages"] = [ToolMessage(content=success_msg, tool_call_id="enhanced_analyze_intent")]
                    
                    self.logger.info("Enhanced intent analysis completed successfully", 
                                   intent=analysis.get("intent_type"),
                                   confidence=analysis.get("confidence"),
                                   tokens=context_data.get("total_tokens"))
                    return state
                    
                except Exception as e:
                    error_msg = f"Enhanced intent analizi sırasında hata oluştu: {str(e)}"
                    state["errors"] = {"error_message": error_msg}
                    state["analysis_result"] = error_msg
                    state["messages"] = [ToolMessage(content=error_msg, tool_call_id="enhanced_analyze_intent")]
                    
                    self.logger.error("Enhanced intent analysis failed", error=str(e))
                    return state
            
            # Add nodes
            graph_builder.add_node("prepare_table_data", prepare_table_data_node)
            graph_builder.add_node("enhanced_analyze_intent", enhanced_analyze_intent_node)
            
            # Define edges
            graph_builder.set_entry_point("prepare_table_data")
            graph_builder.add_edge("prepare_table_data", "enhanced_analyze_intent")
            graph_builder.set_finish_point("enhanced_analyze_intent")
            
            # Compile graph
            compiled_graph = graph_builder.compile(
                checkpointer=self.memory,
                name="enhanced_query_extraction_graph"
            )
            
            self.logger.info("Enhanced Query Extraction graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build Enhanced Query Extraction graph", error=str(e))
            raise
    
    def invoke(self, config: Dict = None) -> Dict:
        """Main entry point for enhanced query extraction"""
        try:
            # Build graph if needed
            if not hasattr(self, '_compiled_graph'):
                self._compiled_graph = self.build_graph()
            
            # Use thread_id for state persistence
            if config is None:
                config = {}
            
            # Ensure thread_id exists for state persistence
            if "configurable" not in config:
                config["configurable"] = {}
            if "thread_id" not in config["configurable"]:
                config["configurable"]["thread_id"] = "enhanced_extraction_session"
            
            # Create initial state
            initial_state = {}
            
            # Execute workflow
            result_state = self._compiled_graph.invoke(initial_state, config)
            
            # Return result
            return {
                "success": "errors" not in result_state or not result_state.get("errors", {}).get("error_message"),
                "table_info": result_state.get("table_info", {}),
                "intent_analysis": result_state.get("intent_analysis", {}),
                "context_analysis": result_state.get("context_analysis", {}),
                "message": result_state.get("analysis_result", "İşlem tamamlandı."),
                "errors": result_state.get("errors", {})
            }
            
        except Exception as e:
            self.logger.error("Enhanced Query Extraction workflow failed", error=str(e))
            return {
                "success": False,
                "table_info": {},
                "intent_analysis": {},
                "context_analysis": {},
                "message": f"Sistem hatası: {str(e)}",
                "errors": {"error_message": str(e)}
            }