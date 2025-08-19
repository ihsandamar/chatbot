# src/graphs/erp_sql_chatbot.py
"""
ERP Customer Service SQL Chatbot
Simple, production-ready Q&A system for restaurant management
Based on LangChain SQL documentation approach
"""

import os
import re
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM, State
from src.services.app_logger import log
from src.services.config_loader import ConfigLoader

# Configuration
config = ConfigLoader.load_config("config/text2sql_config.yaml")

# ================================
# STATE DEFINITION
# ================================

class ERPState(TypedDict):
    """State for ERP SQL chatbot - extends standard State"""
    messages: Annotated[List, ...]  # Standard messages field for supervisor compatibility
    question: str           # Kullanıcı sorusu
    query: str             # Üretilen SQL sorgusu
    result: str            # SQL sonucu
    answer: str            # Formatlanmış cevap
    error: Optional[str]   # Hata mesajı
    
    # Additional context
    intent: Optional[str]   # Sorgu amacı (sales, product, staff, etc.)
    time_range: Optional[str]  # Zaman aralığı (today, this_week, etc.)
    filters: Optional[Dict[str, Any]]  # Ek filtreler


class QueryOutput(TypedDict):
    """Generated SQL query structure"""
    query: Annotated[str, ..., "Valid MSSQL query for MasterReportDocument table"]


# ================================
# PROMPT TEMPLATES
# ================================

SYSTEM_MESSAGE = """Sen bir ERP uzmanı ve SQL sorgulama asistanısın.
Restoran/kafe işletmesi için MasterReportDocument tablosundan veri çekiyorsun.

## TABLO YAPISI - MasterReportDocument

### Ana Belge Alanları:
- DocumentId: Belge/Adisyon benzersiz ID'si
- DocumentDate: İşlem tarihi ve saati
- DocumentStatus: Belge durumu (Açık, Kapalı, İptal vb.)
- DocumentTypeId: Belge tipi ID'si
- DocumentTypeName: Belge tipi adı (Fatura Satış, Pos Fişi, İade vb.)
- DocumentTypeIsReturn: İade belgesi mi? (true/false)
- DocumentKindId: Belge türü ID'si  
- DocumentKindName: Belge türü adı (AA, BB, Tahsilat, Zayi, Paket Servis, Kasa Fişi)

### İşletme ve Şube Bilgileri:
- BusinessId: İşletme ID'si
- BusinessName: İşletme adı
- BusinessType: İşletme tipi (Restoran, Kafe vb.)
- BranchId: Şube ID'si
- BranchName: Şube adı (Elvankent, Çankaya, Batıkent vb.)
- BranchGroupName: Şube grup adı
- BranchCityName: Şubenin bulunduğu şehir
- BranchTownName: Şubenin bulunduğu ilçe

### Ürün ve Satış Detayları:
- DocumentDetailId: Satış satırı detay ID'si
- ProductId: Ürün ID'si
- ProductName: Ürün adı
- CategoryId: Kategori ID'si
- CategoryName: Kategori adı (Kahve, Tatlı, Yemek vb.)
- ProductUnitId: Ürün birim ID'si
- ProductUnitName: Ürün birim adı (Adet, Porsiyon, Bardak vb.)
- Quantity: Satılan miktar
- RowAmount: Satır tutarı (miktar x birim fiyat)
- DiscountAmountTotal: Toplam indirim tutarı
- PurchasePriceWithTax: Vergi dahil alış fiyatı (maliyet)
- DocumentDetailDescription1: Satır açıklaması/notu

### Personel Bilgileri:
- CashierId: Kasiyer/garson ID'si
- CashierName: Kasiyer/garson adı
- DeliveryUserId: Teslimat personeli ID'si
- DeliveryUserName: Teslimat personeli adı (kurye)
- DocumentDetailUpdatedUserId: Son güncelleyen kullanıcı ID'si
- DocumentDetailUpdatedUserFirstLastName: Son güncelleyen kullanıcı adı

### Masa ve Müşteri Bilgileri:
- IsTableActive: Masa aktif mi? (true/false)
- TableNo: Masa numarası
- TableCustomerNumber: Masadaki müşteri sayısı
- CustomerId: Müşteri ID'si
- CustomerName: Müşteri adı

### Zaman Bilgileri:
- DayOfWeek: Haftanın günü (1=Pazartesi, 7=Pazar)
- Hour: İşlem saati (0-23)
- DocumentDetailUpdatedAt: Son güncelleme zamanı

### Özel Durumlar:
- DocumentDetailIsTreatWasteCancel: Özel durum kodu
  - NULL: Normal satış (varsayılan)
  - 1: İkram
  - 2: Zayi
  - 3: İptal

## SQL SORGU KURALLARI:

1. **Tarih Sorguları:**
   - Bugün: WHERE CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
   - Dün: WHERE CAST(DocumentDate AS DATE) = CAST(DATEADD(DAY, -1, GETDATE()) AS DATE)
   - Bu hafta: WHERE DocumentDate >= DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0)
   - Bu ay: WHERE MONTH(DocumentDate) = MONTH(GETDATE()) AND YEAR(DocumentDate) = YEAR(GETDATE())

2. **Satış Sorguları:**
   - Normal satışlar için: WHERE DocumentTypeIsReturn = 0 AND DocumentDetailIsTreatWasteCancel IS NULL
   - İadeler için: WHERE DocumentTypeIsReturn = 1
   - İkramlar için: WHERE DocumentDetailIsTreatWasteCancel = 1
   - Zayi için: WHERE DocumentDetailIsTreatWasteCancel = 2
   - İptal edilenler için: WHERE DocumentDetailIsTreatWasteCancel = 3
   - Paket servis: WHERE DocumentKindName = 'Paket Servis'
   - Masa servisi: WHERE DocumentKindName IN ('AA', 'BB') AND IsTableActive = 1
   - Özel durumlar hariç: WHERE DocumentDetailIsTreatWasteCancel IS NULL

3. **Agregasyon Fonksiyonları:**
   - Toplam satış (özel durumlar hariç): SUM(CASE WHEN DocumentDetailIsTreatWasteCancel IS NULL THEN RowAmount - DiscountAmountTotal ELSE 0 END)
   - Net tutar: SUM(RowAmount) - SUM(DiscountAmountTotal)
   - Ortalama sepet: AVG(RowAmount)
   - İşlem sayısı: COUNT(DISTINCT DocumentId)
   - Ürün adedi: SUM(Quantity)
   - Kar marjı: SUM(RowAmount - PurchasePriceWithTax)
   - İkram tutarı: SUM(CASE WHEN DocumentDetailIsTreatWasteCancel = 1 THEN RowAmount ELSE 0 END)
   - Zayi tutarı: SUM(CASE WHEN DocumentDetailIsTreatWasteCancel = 2 THEN RowAmount ELSE 0 END)
   - İptal tutarı: SUM(CASE WHEN DocumentDetailIsTreatWasteCancel = 3 THEN RowAmount ELSE 0 END)

4. **Gruplama:**
   - Şube bazlı: GROUP BY BranchName
   - Kategori bazlı: GROUP BY CategoryName
   - Ürün bazlı: GROUP BY ProductName
   - Personel bazlı: GROUP BY CashierName
   - Günlük: GROUP BY CAST(DocumentDate AS DATE)
   - Saatlik: GROUP BY Hour
   - Müşteri bazlı: GROUP BY CustomerName

5. **Performans İçin:**
   - Her zaman TOP veya LIMIT kullan (varsayılan TOP 100)
   - Büyük tarih aralıkları için indeks kullan
   - Gereksiz kolonları çekme
   - ORDER BY ile anlamlı sıralama yap

## ÖRNEK SORGULAR:

-- Bugünün satışları (iadeler, ikram, zayi, iptal hariç)
SELECT 
    COUNT(DISTINCT DocumentId) as AdisyonSayisi,
    SUM(Quantity) as ToplamAdet,
    SUM(RowAmount) as BrutTutar,
    SUM(DiscountAmountTotal) as ToplamIndirim,
    SUM(RowAmount - DiscountAmountTotal) as NetTutar
FROM MasterReportDocument
WHERE CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentTypeIsReturn = 0
    AND DocumentDetailIsTreatWasteCancel IS NULL

-- Şube bazlı satış raporu
SELECT 
    BranchName as Sube,
    COUNT(DISTINCT DocumentId) as IslemSayisi,
    SUM(RowAmount - DiscountAmountTotal) as NetSatis,
    AVG(RowAmount) as OrtalamaFis
FROM MasterReportDocument
WHERE CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentTypeIsReturn = 0
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY BranchName
ORDER BY NetSatis DESC

-- En çok satan ürünler
SELECT TOP 10
    ProductName as Urun,
    CategoryName as Kategori,
    SUM(Quantity) as ToplamAdet,
    SUM(RowAmount - DiscountAmountTotal) as ToplamTutar
FROM MasterReportDocument
WHERE DocumentDate >= DATEADD(DAY, -7, GETDATE())
    AND DocumentTypeIsReturn = 0
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY ProductName, CategoryName
ORDER BY ToplamAdet DESC

-- Kasiyer performansı
SELECT 
    CashierName as Personel,
    COUNT(DISTINCT DocumentId) as IslemSayisi,
    SUM(RowAmount - DiscountAmountTotal) as ToplamSatis,
    AVG(RowAmount) as OrtalamaSatis
FROM MasterReportDocument
WHERE CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentTypeIsReturn = 0
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY CashierName
ORDER BY ToplamSatis DESC

-- Masa durumu (aktif masalar)
SELECT 
    TableNo as MasaNo,
    MAX(TableCustomerNumber) as MusteriSayisi,
    COUNT(DISTINCT DocumentId) as AcikHesap,
    SUM(RowAmount - DiscountAmountTotal) as ToplamTutar
FROM MasterReportDocument
WHERE IsTableActive = 1
    AND CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY TableNo
ORDER BY TableNo

-- Paket servis analizi
SELECT 
    DeliveryUserName as Kurye,
    COUNT(DISTINCT DocumentId) as TeslimatSayisi,
    SUM(RowAmount - DiscountAmountTotal) as ToplamTutar
FROM MasterReportDocument
WHERE DocumentKindName = 'Paket Servis'
    AND CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY DeliveryUserName
ORDER BY TeslimatSayisi DESC

-- Saatlik satış dağılımı
SELECT 
    Hour as Saat,
    COUNT(DISTINCT DocumentId) as IslemSayisi,
    SUM(RowAmount - DiscountAmountTotal) as ToplamSatis
FROM MasterReportDocument
WHERE CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentTypeIsReturn = 0
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY Hour
ORDER BY Hour

-- Kategori bazlı performans
SELECT 
    CategoryName as Kategori,
    COUNT(DISTINCT ProductId) as UrunCesidi,
    SUM(Quantity) as ToplamAdet,
    SUM(RowAmount - DiscountAmountTotal) as ToplamTutar,
    SUM(RowAmount - PurchasePriceWithTax) as KarMarji
FROM MasterReportDocument
WHERE DocumentDate >= DATEADD(MONTH, -1, GETDATE())
    AND DocumentTypeIsReturn = 0
    AND DocumentDetailIsTreatWasteCancel IS NULL
GROUP BY CategoryName
ORDER BY ToplamTutar DESC

-- İkram edilen ürünler
SELECT 
    ProductName as Urun,
    SUM(Quantity) as IkramAdedi,
    SUM(RowAmount) as IkramTutari
FROM MasterReportDocument
WHERE CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)
    AND DocumentDetailIsTreatWasteCancel = 1
GROUP BY ProductName
ORDER BY IkramAdedi DESC

-- Zayi olan ürünler
SELECT 
    ProductName as Urun,
    CategoryName as Kategori,
    SUM(Quantity) as ZayiAdedi,
    SUM(RowAmount) as ZayiTutari
FROM MasterReportDocument
WHERE DocumentDate >= DATEADD(WEEK, -1, GETDATE())
    AND DocumentDetailIsTreatWasteCancel = 2
GROUP BY ProductName, CategoryName
ORDER BY ZayiTutari DESC

-- İptal edilen siparişler
SELECT 
    CAST(DocumentDate AS DATE) as Tarih,
    COUNT(DISTINCT DocumentId) as IptalSayisi,
    SUM(RowAmount) as IptalTutari,
    CashierName as IptalEden
FROM MasterReportDocument
WHERE DocumentDate >= DATEADD(DAY, -7, GETDATE())
    AND DocumentDetailIsTreatWasteCancel = 3
GROUP BY CAST(DocumentDate AS DATE), CashierName
ORDER BY Tarih DESC

Kullanıcı Sorusu: {question}

Yukarıdaki tablo yapısı ve örnekleri kullanarak, kullanıcının sorusuna uygun syntactically correct MSSQL sorgusu üret.
Normal satışlar için DocumentDetailIsTreatWasteCancel IS NULL kullan (ikram, zayi, iptal hariç).
İadeler için DocumentTypeIsReturn = 1, normal satışlar için DocumentTypeIsReturn = 0 kullan.
Net tutarlar için her zaman (RowAmount - DiscountAmountTotal) formülünü kullan."""


ANSWER_TEMPLATE = """Sen yardımcı bir müşteri hizmetleri asistanısın.

SQL sonucunu kullanıcı dostu bir şekilde açıkla.

Kullanıcı Sorusu: {question}
SQL Sorgusu: {query}
SQL Sonucu: {result}

Kurallar:
- Emoji kullan (📊, 💰, 📈, ✅, 🏪)
- Sayıları formatla (₺ para birimi)
- Önemli bilgileri vurgula
- Kısa ve net ol
- Türkçe yanıt ver

Formatlanmış yanıt:"""

# ================================
# CORE FUNCTIONS
# ================================

class ERPQueryAnalyzer:
    """Analyze user queries and extract intent"""
    
    def __init__(self):
        self.logger = log.get(module="erp_analyzer")
        
        # Intent patterns
        self.intent_patterns = {
            "sales": ["satış", "ciro", "hasılat", "kazanç", "gelir"],
            "product": ["ürün", "kategori", "en çok satan", "popüler"],
            "staff": ["personel", "kasiyer", "garson", "çalışan", "performans"],
            "customer": ["müşteri", "customer", "alıcı"],
            "payment": ["ödeme", "nakit", "kart", "payment"],
            "branch": ["şube", "branch", "mağaza"],
            "comparison": ["karşılaştır", "kıyasla", "göre", "fark"],
            "time_based": ["bugün", "dün", "hafta", "ay", "tarih"]
        }
        
        # Time patterns
        self.time_patterns = {
            "today": ["bugün", "today"],
            "yesterday": ["dün", "yesterday"],
            "this_week": ["bu hafta", "this week"],
            "last_week": ["geçen hafta", "last week"],
            "this_month": ["bu ay", "this month"],
            "last_month": ["geçen ay", "last month"]
        }
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Analyze question and extract metadata"""
        question_lower = question.lower()
        
        # Detect intent
        intent = "general"
        for key, patterns in self.intent_patterns.items():
            if any(p in question_lower for p in patterns):
                intent = key
                break
        
        # Detect time range
        time_range = "today"  # Default
        for key, patterns in self.time_patterns.items():
            if any(p in question_lower for p in patterns):
                time_range = key
                break
        
        # Extract filters
        filters = {}
        
        # Branch filter
        branch_match = re.search(r'(elvankent|çankaya|batıkent|ümitköy)', question_lower)
        if branch_match:
            filters["branch"] = branch_match.group(1).title()
        
        # Payment type filter
        if "nakit" in question_lower:
            filters["payment_type"] = "Nakit"
        elif "kart" in question_lower or "kredi" in question_lower:
            filters["payment_type"] = "Kredi Kartı"
        
        self.logger.info("Query analyzed", 
                        intent=intent, 
                        time_range=time_range,
                        filters=filters)
        
        return {
            "intent": intent,
            "time_range": time_range,
            "filters": filters
        }


def write_query(state: ERPState, llm: LLM, analyzer: ERPQueryAnalyzer) -> ERPState:
    """Generate SQL query from question"""
    logger = log.get(module="write_query")
    
    try:
        # Analyze question
        analysis = analyzer.analyze(state["question"])
        state["intent"] = analysis["intent"]
        state["time_range"] = analysis["time_range"]
        state["filters"] = analysis.get("filters", {})
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            ("human", "{question}")
        ])
        
        # Generate SQL with structured output
        structured_llm = llm.get_chat().with_structured_output(QueryOutput)
        
        message = prompt.invoke({"question": state["question"]})
        result = structured_llm.invoke(message)
        
        # Clean and validate SQL
        sql = result["query"]
        sql = clean_sql(sql)
        
        # Add safety checks
        if not is_safe_query(sql):
            raise ValueError("Unsafe SQL query detected")
        
        state["query"] = sql
        logger.info("SQL query generated", sql_length=len(sql))
        
    except Exception as e:
        logger.error("Failed to generate query", error=str(e))
        state["query"] = generate_fallback_query(state)
        state["error"] = str(e)
    
    return state


def execute_query(state: ERPState, db: SQLDatabase) -> ERPState:
    """Execute SQL query safely"""
    logger = log.get(module="execute_query")
    
    try:
        # Create execution tool
        execute_tool = QuerySQLDatabaseTool(db=db)
        
        # Execute query
        result = execute_tool.invoke(state["query"])
        
        # Process result
        if result:
            state["result"] = result
            logger.info("Query executed successfully", 
                       result_length=len(str(result)))
        else:
            state["result"] = "Sonuç bulunamadı"
            logger.warning("Query returned no results")
            
    except Exception as e:
        logger.error("Query execution failed", 
                    error=str(e),
                    query=state["query"])
        
        # Try to provide helpful error message
        if "Invalid column name" in str(e):
            state["error"] = "Kolon adı hatası. Tablo yapısını kontrol ediyorum..."
        elif "syntax" in str(e).lower():
            state["error"] = "SQL syntax hatası"
        else:
            state["error"] = f"Sorgu hatası: {str(e)}"
        
        state["result"] = f"HATA: {str(e)}"
    
    return state


def generate_answer(state: ERPState, llm: LLM) -> ERPState:
    """Generate user-friendly answer from SQL results"""
    logger = log.get(module="generate_answer")
    
    try:
        # Check for errors
        if state.get("error"):
            state["answer"] = format_error_message(state["error"], state["question"])
            return state
        
        # Create answer prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_TEMPLATE),
            ("human", "Yanıtı oluştur")
        ])
        
        # Generate answer
        message = prompt.invoke({
            "question": state["question"],
            "query": state["query"],
            "result": state["result"]
        })
        
        response = llm.get_chat().invoke(message)
        
        # Format the answer
        answer = response.content
        
        # Add metadata
        answer = add_metadata_to_answer(answer, state)
        
        state["answer"] = answer
        logger.info("Answer generated", answer_length=len(answer))
        
    except Exception as e:
        logger.error("Failed to generate answer", error=str(e))
        state["answer"] = f"Yanıt oluşturulamadı: {str(e)}"
    
    return state


# ================================
# HELPER FUNCTIONS
# ================================

def clean_sql(sql: str) -> str:
    """Clean and format SQL query"""
    # Remove markdown
    sql = re.sub(r'```sql\s*', '', sql)
    sql = re.sub(r'```\s*', '', sql)
    
    # Clean whitespace
    sql = ' '.join(sql.split())
    
    # Ensure it ends properly
    sql = sql.rstrip(';')
    
    return sql.strip()


def is_safe_query(sql: str) -> bool:
    """Check if SQL query is safe to execute"""
    sql_upper = sql.upper()
    
    # Must be SELECT query
    if not sql_upper.strip().startswith('SELECT'):
        return False
    
    # No dangerous operations
    dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']
    
    for op in dangerous:
        if op in sql_upper:
            return False
    
    return True


def generate_fallback_query(state: ERPState) -> str:
    """Generate a safe fallback query"""
    time_condition = get_time_condition(state.get("time_range", "today"))
    
    return f"""
    SELECT TOP 10
        CAST(DocumentDate AS DATE) as Tarih,
        COUNT(*) as IslemSayisi,
        SUM(DocumentAmount) as ToplamTutar,
        AVG(DocumentAmount) as OrtalamaTutar
    FROM MasterReportDocument
    WHERE {time_condition}
        AND DocumentType = 'SALES'
    GROUP BY CAST(DocumentDate AS DATE)
    ORDER BY Tarih DESC
    """.strip()


def get_time_condition(time_range: str) -> str:
    """Get SQL time condition for given range"""
    conditions = {
        "today": "CAST(DocumentDate AS DATE) = CAST(GETDATE() AS DATE)",
        "yesterday": "CAST(DocumentDate AS DATE) = CAST(DATEADD(DAY, -1, GETDATE()) AS DATE)",
        "this_week": "DocumentDate >= DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0)",
        "last_week": """DocumentDate >= DATEADD(WEEK, -1, DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0))
                        AND DocumentDate < DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0)""",
        "this_month": "MONTH(DocumentDate) = MONTH(GETDATE()) AND YEAR(DocumentDate) = YEAR(GETDATE())",
        "last_month": """MONTH(DocumentDate) = MONTH(DATEADD(MONTH, -1, GETDATE())) 
                         AND YEAR(DocumentDate) = YEAR(DATEADD(MONTH, -1, GETDATE()))"""
    }
    
    return conditions.get(time_range, conditions["today"])


def format_error_message(error: str, question: str) -> str:
    """Format error message for user"""
    return f"""
❌ **İşlem Başarısız**

**Sorunuz:** {question}
**Hata:** {error}

🔄 **Öneriler:**
• Sorunuzu daha basit ifade edin
• Tarih formatını kontrol edin (bugün, dün, bu hafta gibi)
• Şube adlarını doğru yazdığınızdan emin olun

📞 Yardım için destek ekibi ile iletişime geçebilirsiniz.
"""


def add_metadata_to_answer(answer: str, state: ERPState) -> str:
    """Add metadata footer to answer"""
    current_time = datetime.now().strftime("%H:%M")
    
    metadata = f"""

---
📌 **Bilgi**
• Zaman: {current_time}
• Veri: MasterReportDocument
"""
    
    if state.get("intent"):
        metadata += f"• Tip: {state['intent']}\n"
    
    return answer + metadata


def parse_sql_result(result: str) -> List[Dict]:
    """Parse SQL result string into structured data"""
    try:
        # Simple parsing for [(row1), (row2), ...] format
        import ast
        parsed = ast.literal_eval(result)
        
        if isinstance(parsed, list):
            return [{"row": i, "data": row} for i, row in enumerate(parsed)]
    except:
        pass
    
    return []


# ================================
# MAIN GRAPH
# ================================

@register_graph("erp_sql_chatbot")
class ERPSQLChatbot(BaseGraph):
    """Simple and effective ERP SQL Chatbot"""
    
    def __init__(self, llm: LLM, db: SQLDatabase = None):
        super().__init__(llm=llm, state_class=State)  # Use standard State for supervisor compatibility
        self.logger = log.get(module="erp_sql_chatbot")
        
        # Database setup
        if db is None:
            db = SQLDatabase.from_uri(config.database.uri)
        self.db = db
        
        # Components
        self.analyzer = ERPQueryAnalyzer()
        self.memory = MemorySaver()
        
        # Validate database
        self._validate_database()
        
        self.logger.info("ERP SQL Chatbot initialized")
    
    def _validate_database(self):
        """Validate database connection and table"""
        try:
            tables = self.db.get_table_names()
            
            if "MasterReportDocument" not in tables:
                self.logger.warning("MasterReportDocument table not found", 
                                  available_tables=tables)
            else:
                # Get table info
                info = self.db.get_table_info(["MasterReportDocument"])
                self.logger.info("Database validated", 
                               table_info_length=len(info))
                
        except Exception as e:
            self.logger.error("Database validation failed", error=str(e))
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        self.logger.info("Building ERP SQL graph")
        
        # Create graph using standard State
        graph = StateGraph(State)
        
        # Define nodes with wrapped functions that handle state conversion
        def write_query_node(state: State) -> State:
            # Convert State to ERPState
            erp_state = self._convert_to_erp_state(state)
            result = write_query(erp_state, self.llm, self.analyzer)
            # Convert back to State
            return self._convert_to_standard_state(result, state)
        
        def execute_query_node(state: State) -> State:
            # Convert State to ERPState  
            erp_state = self._convert_to_erp_state(state)
            result = execute_query(erp_state, self.db)
            # Convert back to State
            return self._convert_to_standard_state(result, state)
        
        def generate_answer_node(state: State) -> State:
            # Convert State to ERPState
            erp_state = self._convert_to_erp_state(state)
            result = generate_answer(erp_state, self.llm)
            # Convert back to State and add AI message
            new_state = self._convert_to_standard_state(result, state)
            # Add AI response message to ensure supervisor gets the response
            if result.get("answer"):
                # Ensure messages is a list
                current_messages = new_state.get("messages", [])
                if not isinstance(current_messages, list):
                    current_messages = []
                
                # Log the result for debugging
                self.logger.info("Generated answer for supervisor", 
                               answer_preview=result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"])
                
                new_state["messages"] = current_messages + [AIMessage(content=result["answer"])]
            else:
                # Fallback if no answer was generated
                self.logger.warning("No answer generated, adding fallback response")
                current_messages = new_state.get("messages", [])
                if not isinstance(current_messages, list):
                    current_messages = []
                new_state["messages"] = current_messages + [AIMessage(content="Yanıt oluşturulamadı.")]
            return new_state
        
        # Add nodes
        graph.add_node("write_query", write_query_node)
        graph.add_node("execute_query", execute_query_node)
        graph.add_node("generate_answer", generate_answer_node)
        
        # Define flow
        graph.add_edge(START, "write_query")
        graph.add_edge("write_query", "execute_query")
        graph.add_edge("execute_query", "generate_answer")
        graph.add_edge("generate_answer", END)
        
        # Compile with memory (for human-in-the-loop if needed)
        compiled = graph.compile(
            name="master_report_graph",
            checkpointer=self.memory,
            interrupt_before=[]  # Can add "execute_query" for human approval
        )
        
        self.logger.info("Graph compiled successfully")
        return compiled
    
    def _convert_to_erp_state(self, state: State) -> ERPState:
        """Convert standard State to ERPState"""
        # Extract question from messages
        question = ""
        if state.get("messages"):
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    question = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    question = msg.get("content", "")
                    break
        
        return ERPState(
            messages=state.get("messages", []),
            question=question,
            query=state.get("generated_sql", ""),
            result=state.get("sql_result", ""),
            answer="",
            error=state.get("error_message"),
            intent=state.get("detected_intent"),
            time_range=None,
            filters=None
        )
    
    def _convert_to_standard_state(self, erp_state: ERPState, original_state: State) -> State:
        """Convert ERPState back to standard State"""
        # Ensure messages is properly carried forward
        messages = original_state.get("messages", [])
        if not isinstance(messages, list):
            messages = []
            
        new_state = State(
            messages=messages,
            generated_sql=erp_state.get("query", ""),
            sql_result=erp_state.get("result", ""),
            error_message=erp_state.get("error"),
            detected_intent=erp_state.get("intent")
        )
        return new_state
    
    def invoke(self, messages: List[Any], config: Optional[Dict] = None) -> str:
        """Main entry point - process user message"""
        try:
            # Extract question from messages
            question = self._extract_question(messages)
            
            if not question:
                return self._create_help_message()
            
            # Create initial state
            initial_state = ERPState(
                question=question,
                query="",
                result="",
                answer="",
                error=None,
                intent=None,
                time_range=None,
                filters=None
            )
            
            # Build graph if needed
            if not hasattr(self, '_compiled_graph'):
                self._compiled_graph = self.build_graph()
            
            # Process through graph
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
            
            # Run the graph
            result = self._compiled_graph.invoke(initial_state, config)
            
            # Return the answer
            return result.get("answer", "Yanıt oluşturulamadı")
            
        except Exception as e:
            self.logger.error("Processing failed", error=str(e))
            return self._create_error_message(str(e))
    
    def stream(self, messages: List[Any], config: Optional[Dict] = None):
        """Stream processing steps"""
        try:
            question = self._extract_question(messages)
            
            initial_state = ERPState(
                question=question,
                query="",
                result="",
                answer="",
                error=None,
                intent=None,
                time_range=None,
                filters=None
            )
            
            if not hasattr(self, '_compiled_graph'):
                self._compiled_graph = self.build_graph()
            
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
            
            # Stream updates
            for update in self._compiled_graph.stream(
                initial_state, 
                config, 
                stream_mode="updates"
            ):
                yield update
                
        except Exception as e:
            self.logger.error("Streaming failed", error=str(e))
            yield {"error": str(e)}
    
    def _extract_question(self, messages: List[Any]) -> str:
        """Extract question from messages"""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
                elif isinstance(msg, HumanMessage):
                    return msg.content
                elif hasattr(msg, "type") and msg.type == "human":
                    return msg.content if hasattr(msg, "content") else ""
        
        return ""
    
    def _create_help_message(self) -> str:
        """Create help message"""
        return """
🤖 **ERP Müşteri Hizmetleri Asistanı**

Size nasıl yardımcı olabilirim? İşte yapabileceklerim:

📊 **Satış Raporları**
• "Bugünün satışları ne kadar?"
• "Bu haftanın cirosu?"
• "Elvankent şubesinin satışları?"

👥 **Personel Performansı**
• "Hangi kasiyer en çok satış yaptı?"
• "Personel bazlı satış raporu"

💰 **Ödeme Analizleri**
• "Nakit ve kart ödemelerinin dağılımı?"
• "En çok hangi ödeme tipi kullanılıyor?"

📈 **Karşılaştırmalar**
• "Bu ay geçen aya göre nasıl?"
• "Hafta içi ve hafta sonu karşılaştırması"

🕐 **Zaman Bazlı Sorgular**
• "Dünün satışları"
• "Geçen haftanın özeti"
• "Bu ayın performansı"

Lütfen sorunuzu yazın...
"""
    
    def _create_error_message(self, error: str) -> str:
        """Create error message"""
        return f"""
❌ **Sistem Hatası**

Üzgünüm, bir hata oluştu:
{error}

🔄 Lütfen tekrar deneyin veya sorunuzu basitleştirin.

💡 **İpucu:** Tarih ifadelerini (bugün, dün, bu hafta) ve basit sorgular kullanın.
"""


# ================================
# EXAMPLE USAGE
# ================================

def example_usage():
    """Example usage of ERP SQL Chatbot"""
    
    # Initialize
    from src.models.models import LLM
    
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    chatbot = ERPSQLChatbot(llm)
    
    # Test queries
    test_queries = [
        "Bugünün satışları ne kadar?",
        "Hangi kasiyer en çok satış yaptı?",
        "Elvankent şubesinin bu haftaki performansı?",
        "Nakit ve kart ödemelerinin dağılımı nasıl?",
        "En çok satan ürünler hangileri?",
        "Bu ay geçen aya göre nasıl?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Soru: {query}")
        
        messages = [{"role": "user", "content": query}]
        answer = chatbot.invoke(messages)
        
        print(f"🤖 Yanıt: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    example_usage()