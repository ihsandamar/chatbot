# src/graphs/text2sql_graph.py - SOLID Refactored Version
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import State
from src.tools.langgraph_sql_tools import LangGraphSQLTools
from src.tools.custom_sql_tools import CustomSQLTools

from langchain_community.utilities import SQLDatabase
from src.services.config_loader import ConfigLoader


config = ConfigLoader.load_config("config/text2sql_config.yaml")


# SOLID Principle 1: Single Responsibility Principle (SRP)
class MessageExtractor:
    """Mesajlardan bilgi çıkarma sorumluluğu - SRP"""
    
    @staticmethod
    def get_last_user_question(messages: List) -> str:
        """Son kullanıcı sorusunu çıkar"""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list) and len(msg.content) > 0:
                        if isinstance(msg.content[0], dict) and 'text' in msg.content[0]:
                            return msg.content[0]['text']
                    elif isinstance(msg.content, str):
                        return msg.content
        return "Soru bulunamadı"
    
    @staticmethod
    def find_tool_result(messages: List, tool_call_pattern: str) -> tuple[str, str]:
        """Tool sonucunu ve çalıştırılan query'yi bul"""
        query_result = ""
        executed_query = ""
        
        for msg in reversed(messages):
            if hasattr(msg, 'tool_call_id') and hasattr(msg, 'content'):
                if tool_call_pattern in str(msg.tool_call_id):
                    query_result = msg.content
                    
                    # Önceki AI message'da query var
                    msg_index = messages.index(msg)
                    if msg_index > 0:
                        prev_msg = messages[msg_index - 1]
                        if hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for tc in prev_msg.tool_calls:
                                if tc.get("name") == "db_query_tool":
                                    executed_query = tc["args"].get("__arg1", "") or tc["args"].get("query", "")
                                    break
                    break
        
        return query_result, executed_query


class TableSelector:
    """Tablo seçme sorumluluğu - SRP"""
    
    @staticmethod
    def select_relevant_tables(user_question: str, available_tables: List[str]) -> List[str]:
        """Kullanıcı sorusuna göre ilgili tabloları seç"""
        user_question_lower = user_question.lower()
        
        if "product" in user_question_lower or "espresso" in user_question_lower:
            return ["Products", "Categories", "ProductCategories"]
        elif "order" in user_question_lower:
            return ["Orders", "OrderDetails", "Products", "Customers"]
        elif "customer" in user_question_lower:
            return ["Customers", "Orders"]
        else:
            # Default olarak ilk 3 tabloyu al
            return available_tables[:3]
    
    @staticmethod
    def is_master_report_request(user_question: str) -> bool:
        """MasterReport modülü için istek mi kontrol et"""
        master_report_keywords = [
            "ciro raporu", "paket servis raporu", "master report", "masterreport",
            "satış raporu", "ürün raporu", "kategori raporu", "günlük rapor",
            "haftalık rapor", "aylık rapor", "işletme raporu", "şube raporu"
        ]
        
        user_question_lower = user_question.lower()
        return any(keyword in user_question_lower for keyword in master_report_keywords)


# SOLID Principle 2: Open/Closed Principle (OCP)
class NodeInterface(ABC):
    """Node'lar için ortak interface - OCP"""
    
    @abstractmethod
    def execute(self, state: State) -> Dict[str, Any]:
        pass


class BaseNode(NodeInterface):
    """Node'lar için base class - OCP"""
    
    def __init__(self, name: str):
        self.name = name
    
    def log_debug(self, message: str):
        print(f"[DEBUG] {self.name}: {message}")


class ListTablesNode(BaseNode):
    """Tabloları listeleme node'u - SRP"""
    
    def __init__(self):
        super().__init__("ListTablesNode")
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Listing all database tables")
        messages = state["messages"]
        
        response = AIMessage(
            content="Listing all database tables to understand the schema.",
            tool_calls=[{
                "name": "sql_db_list_tables",
                "args": {},
                "id": "list_tables_call",
            }]
        )
        
        return {"messages": messages + [response]}


class GetSchemaNode(BaseNode):
    """Şema alma node'u - SRP"""
    
    def __init__(self):
        super().__init__("GetSchemaNode")
        self.message_extractor = MessageExtractor()
        self.table_selector = TableSelector()
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Getting table schemas")
        messages = state["messages"]
        
        # Table listesini bul
        table_list = ""
        for msg in reversed(messages):
            if hasattr(msg, 'tool_call_id') and "list_tables" in str(msg.tool_call_id):
                table_list = msg.content
                break
        
        if table_list:
            # Kullanıcı sorusunu al ve ilgili tabloları seç
            user_question = self.message_extractor.get_last_user_question(messages)
            available_tables = [t.strip() for t in table_list.replace('[', '').replace(']', '').split(',')]
            important_tables = self.table_selector.select_relevant_tables(user_question, available_tables)
            
            self.log_debug(f"Selected tables: {important_tables}")
            
            response = AIMessage(
                content=f"Getting schema for relevant tables: {', '.join(important_tables)}",
                tool_calls=[{
                    "name": "sql_db_schema",
                    "args": {"table_names": ", ".join(important_tables)},
                    "id": "schema_call",
                }]
            )
            
            return {"messages": messages + [response]}
        
        # Table list yoksa boş response
        response = AIMessage(content="No tables found to get schema.")
        return {"messages": messages + [response]}


class ExecuteQueryNode(BaseNode):
    """Query çalıştırma node'u - SRP"""
    
    def __init__(self, llm, tools):
        super().__init__("ExecuteQueryNode")
        self.query_model = self._create_query_model(llm, tools)
        self.master_report_query_model = self._create_master_report_query_model(llm, tools)
    
    def _create_query_model(self, llm, tools):
        """Normal query model oluştur"""
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MSSQL expert. Based on the table schemas provided, generate a syntactically correct MSSQL query and execute it.

Rules:
- Use the exact table and column names from the schemas
- Join tables properly using foreign keys
- Get at most 5 results unless specified otherwise
- Only query relevant columns
- Use meaningful ORDER BY clauses
- Handle many-to-many relationships through junction tables

After generating the SQL query, IMMEDIATELY use db_query_tool to execute it."""),
            ("placeholder", "{messages}")
        ])
        
        return query_prompt | llm.bind_tools([
            tool for tool in tools if tool.name == "db_query_tool"
        ], tool_choice="required")
    
    def _create_master_report_query_model(self, llm, tools):
        """MasterReport query model oluştur"""
        master_report_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MSSQL expert for MasterReport module. The MasterReport temp tables have been created with the following structure:

#MasterReportTable contains columns:
- İşletme_Adı, Şube_Adı, Kasa_Adı, Masa_Adı, Cari_Adı
- Şube_Grup_Adı, İşletme_Tipi, Şube_Şehir_Adı, Şube_İlçe_Adı
- Adisyon_Id, Adisyon_Tarihi, Belge_Nevi_Adı, Belge_Tipi_Adı
- Iade_Belgesi_mi, Haftanın_Günü, Saat
- Adisyon_Detay_Id, Ürün_Adı, Miktar, Ürün_Tutarı
- Ürüne_Uygulanan_İndirim_Tutarı, Ürün_Kategori_Adı, Ürün_Birim_Adı
- Ikram_Zayi_İptal_Mi

Generate reports using #MasterReportTable for:
- Ciro raporu (Revenue reports)
- Paket servis raporu (Package service reports)
- Satış raporu (Sales reports)
- Ürün raporu (Product reports)
- Kategori raporu (Category reports)

Use Turkish column names as shown above. Generate appropriate GROUP BY, SUM, COUNT queries.
After generating the SQL query, IMMEDIATELY use db_query_tool to execute it."""),
            ("placeholder", "{messages}")
        ])
        
        return master_report_prompt | llm.bind_tools([
            tool for tool in tools if tool.name == "db_query_tool"
        ], tool_choice="required")
    
    def execute(self, state: State) -> Dict[str, Any]:
        messages = state["messages"]
        
        # MasterReport setup yapıldı mı kontrol et
        master_report_setup = any(
            hasattr(msg, 'tool_call_id') and "master_report_setup" in str(msg.tool_call_id) 
            for msg in messages
        )
        
        # Schema var mı kontrol et (normal flow için)
        schema_found = any(
            hasattr(msg, 'tool_call_id') and "schema" in str(msg.tool_call_id) 
            for msg in messages
        )
        
        if master_report_setup:
            self.log_debug("Generate MasterReport query")
            response = self.master_report_query_model.invoke({"messages": messages})
            self.log_debug(f"MasterReport query model response with {len(getattr(response, 'tool_calls', []))} tool calls")
            return {"messages": messages + [response]}
        elif schema_found:
            self.log_debug("Generate normal query")
            response = self.query_model.invoke({"messages": messages})
            self.log_debug(f"Normal query model response with {len(getattr(response, 'tool_calls', []))} tool calls")
            return {"messages": messages + [response]}
        else:
            response = AIMessage(content="Schema or MasterReport setup not found, cannot generate query.")
            return {"messages": messages + [response]}


class MasterReportNode(BaseNode):
    """MasterReport temp tabloları oluşturma node'u - SRP"""
    
    def __init__(self):
        super().__init__("MasterReportNode")
        self.master_report_sql = self._get_master_report_sql()
    
    def _get_master_report_sql(self) -> str:
        """MasterReport için hazır SQL kodunu döndür"""
        return """
        -- MasterReport Query Merged
        SELECT  
            D.Id AS DocumentId,
            D.BranchId,
            D.BusinessId,
            D.DocumentTypeId,
            D.DocumentKindId,
            D.CashierId,
            D.UserId,
            D.Status AS DocumentStatus,
            D.Date,
            D.SpecialCode1 COLLATE SQL_Latin1_General_CP1_CI_AS AS TableNo,
            D.CustomerId,
            D.CustomerName COLLATE SQL_Latin1_General_CP1_CI_AS AS CustomerName,
            D.SpecialCode3 COLLATE SQL_Latin1_General_CP1_CI_AS AS TableCustomerNumber,
            0 AS IsTableActive, 
            
            --DocumentDetails
            DD.Id AS DocumentDetailId,
            DD.ProductId,
            DD.WaybillId,
            DD.BillId,
            DD.VoucherId,
            ISNULL(DD.Quantity, 0) AS Quantity,
            ISNULL(DD.RowAmount, 0) AS RowAmount,
            ISNULL(DD.DiscountAmountTotal, 0) AS DiscountAmountTotal, 
            DD.IsTreatWasteCancel, 
            DD.ProductUnitId, 
            DD.UpdatedAt, 
            DD.UpdatedUserId, 
            DATEPART(DW, DD.Maturity) AS DayOfWeek, 
            DATEPART(HOUR, DD.Maturity) AS Hour, 
            DD.PurchasePriceWithTax,
            DD.Description1,
            
            --DocumentTypes
            DT.Name AS DocumentTypeName,
            DT.IsReturn AS DocumentTypeIsReturn
        INTO #Documents
        FROM 
            Documents D WITH (NOLOCK)
            LEFT JOIN 
                DocumentDetails DD ON 
                    CASE 
                        WHEN ISNULL(DD.WaybillId, '00000000-0000-0000-0000-000000000000') = '00000000-0000-0000-0000-000000000000' 
                             AND ISNULL(DD.BillId, '00000000-0000-0000-0000-000000000000') = '00000000-0000-0000-0000-000000000000'
                        THEN ISNULL(DD.VoucherId, '00000000-0000-0000-0000-000000000000') 
                        WHEN ISNULL(DD.BillId, '00000000-0000-0000-0000-000000000000') = '00000000-0000-0000-0000-000000000000' 
                        THEN ISNULL(DD.WaybillId, '00000000-0000-0000-0000-000000000000') 
                        ELSE ISNULL(DD.BillId, '00000000-0000-0000-0000-000000000000') 
                    END = D.Id
            LEFT JOIN 
                DocumentTypes DT ON D.DocumentTypeId = DT.Id 
        WHERE 
            D.Status = 1
            AND DD.Status = 1
            AND DT.Status = 1
            AND DT.JoinType IN (1, 2, 3);

        SELECT  
            D.Id AS DocumentId,
            D.BranchId,
            D.BusinessId,
            D.DocumentTypeId,
            D.DocumentKindId,
            D.CashierId,
            D.UserId,
            D.Status AS DocumentStatus,
            D.Date,
            D.SpecialCode1 COLLATE SQL_Latin1_General_CP1_CI_AS AS TableNo,
            D.CustomerId,
            D.CustomerName COLLATE SQL_Latin1_General_CP1_CI_AS AS CustomerName,
            D.SpecialCode3 COLLATE SQL_Latin1_General_CP1_CI_AS AS TableCustomerNumber,
            D.IsTableActive, 
            
            --TempDocumentDetails
            DD.Id AS DocumentDetailId,
            DD.ProductId,
            DD.WaybillId,
            DD.BillId,
            DD.VoucherId,
            ISNULL(DD.Quantity, 0) AS Quantity,
            ISNULL(DD.RowAmount, 0) AS RowAmount,
            ISNULL(DD.DiscountAmountTotal, 0) AS DiscountAmountTotal, 
            DD.IsTreatWasteCancel, 
            DD.ProductUnitId, 
            DD.UpdatedAt, 
            DD.UpdatedUserId, 
            DATEPART(DW, DD.Maturity) AS DayOfWeek, 
            DATEPART(HOUR, DD.Maturity) AS Hour, 
            DD.PurchasePriceWithTax,
            DD.Description1,
            
            --DocumentTypes
            CAST(NULL AS NVARCHAR(100)) AS DocumentTypeName,
            CAST(NULL AS BIT) AS DocumentTypeIsReturn
        INTO #TempDocuments
        FROM 
            TempDocuments D WITH (NOLOCK)
        LEFT JOIN 
            TempDocumentDetails DD ON 
                CASE 
                    WHEN ISNULL(WaybillId, '00000000-0000-0000-0000-000000000000') = '00000000-0000-0000-0000-000000000000' 
                         AND ISNULL(BillId, '00000000-0000-0000-0000-000000000000') = '00000000-0000-0000-0000-000000000000'
                    THEN ISNULL(VoucherId, '00000000-0000-0000-0000-000000000000') 
                    WHEN ISNULL(BillId, '00000000-0000-0000-0000-000000000000') = '00000000-0000-0000-0000-000000000000' 
                    THEN ISNULL(WaybillId, '00000000-0000-0000-0000-000000000000') 
                    ELSE ISNULL(BillId, '00000000-0000-0000-0000-000000000000') 
                END = D.Id
        WHERE 1=1;

        SELECT * INTO #UnionDocuments FROM (
            SELECT * FROM #Documents
            UNION ALL
            SELECT * FROM #TempDocuments
        ) AS CombinedData;

        SELECT  
            Id, 
            CONCAT(FirstName, ' ', LastName) AS FirstLastName 
        INTO #Users
        FROM 
            Users WITH (NOLOCK)
        WHERE 1=1 
            AND Status = 1;

        SELECT  
            P.Id AS ProductId, 
            P.Name AS ProductName, 
            C.Id AS CategoryId, 
            C.Name AS CategoryName 
        INTO #ProductCategory
        FROM 
            Products P WITH (NOLOCK)
        LEFT JOIN 
            ProductCategories PC WITH (NOLOCK) ON P.Id = PC.ProductId
        LEFT JOIN 
            Categories C WITH (NOLOCK) ON PC.CategoryId = C.Id
        WHERE 
            1 = 1 
            AND PC.Status = 1 
            AND P.Status = 1
            AND C.Status = 1;

        SELECT 
            --İşletme - şube
            BS.Name AS İşletme_Adı,
            B.Name AS Şube_Adı,
            CS.Name AS Kasa_Adı,
            D.TableNo AS Masa_Adı,
            D.CustomerName AS Cari_Adı,
            IB.GroupName AS Şube_Grup_Adı,
            IB.BusinessType AS İşletme_Tipi,
            CTY.Name AS Şube_Şehir_Adı,
            TWN.Name AS Şube_İlçe_Adı,
            
            --Adisyon Bilgileri
            D.DocumentId AS Adisyon_Id,
            D.Date AS Adisyon_Tarihi,
            DK.Name AS Belge_Nevi_Adı,
            D.DocumentTypeName AS Belge_Tipi_Adı,
            D.DocumentTypeIsReturn AS Iade_Belgesi_mi,
            D.DayOfWeek AS Haftanın_Günü,
            D.Hour AS Saat,
            
            --Adisyon Detay Bilgileri
            D.DocumentDetailId AS Adisyon_Detay_Id,
            PC.ProductName AS Ürün_Adı,
            D.Quantity AS Miktar,
            D.RowAmount AS Ürün_Tutarı,
            D.DiscountAmountTotal AS Ürüne_Uygulanan_İndirim_Tutarı,
            PC.CategoryName AS Ürün_Kategori_Adı,
            UN.Name AS Ürün_Birim_Adı,
            D.IsTreatWasteCancel AS Ikram_Zayi_İptal_Mi
        INTO #MasterReportTable
        FROM 
            #UnionDocuments D
        LEFT JOIN 
            Branches B ON D.BranchId = B.Id AND B.Status = 1
        LEFT JOIN 
            #ProductCategory PC ON D.ProductId = PC.ProductId
        LEFT JOIN 
            DocumentKinds DK ON D.DocumentKindId = DK.Id AND DK.Status = 1
        LEFT JOIN 
            #Users U ON D.UserId = U.Id
        LEFT JOIN 
            ProductUnits PU ON D.ProductUnitId = PU.Id AND PU.Status = 1
        LEFT JOIN 
            Units UN ON PU.UnitId = UN.Id AND UN.Status = 1
        LEFT JOIN 
            Cashiers CS ON D.CashierId = CS.Id AND CS.Status = 1
        LEFT JOIN 
            IntegrationBranches IB ON B.Id = IB.DefaultBranchId AND IB.Status = 1
        LEFT JOIN 
            Cities CTY ON B.CityId = CTY.Id AND CTY.Status = 1
        LEFT JOIN 
            Towns TWN ON B.TownId = TWN.Id AND TWN.Status = 1
        LEFT JOIN 
            #Users DDUpdetedUser ON D.UpdatedUserId = DDUpdetedUser.Id
        LEFT JOIN 
            Businesses BS ON D.BusinessId = BS.Id AND BS.Status = 1
        WHERE 1=1;
        """
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Creating MasterReport temp tables")
        messages = state["messages"]
        
        response = AIMessage(
            content="Creating MasterReport temp tables for advanced reporting.",
            tool_calls=[{
                "name": "db_query_tool",
                "args": {"query": self.master_report_sql},
                "id": "master_report_setup_call",
            }]
        )
        
        return {"messages": messages + [response]}


class FixQueryNode(BaseNode):
    """SQL hata düzeltme node'u - SRP"""
    
    def __init__(self, llm, tools):
        super().__init__("FixQueryNode")
        self.fix_query_model = self._create_fix_query_model(llm, tools)
    
    def _create_fix_query_model(self, llm, tools):
        """Query düzeltme model oluştur"""
        fix_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MSSQL expert specialized in fixing SQL errors.

You will receive:
1. The original SQL query that failed
2. The error message from the database

Your task:
- Analyze the error message carefully
- Fix the SQL query based on the error
- Common MSSQL errors to watch for:
  * Syntax errors (missing commas, parentheses, etc.)
  * Column name errors (check exact column names)
  * Table name errors (check exact table names) 
  * Data type mismatches
  * JOIN syntax errors
  * WHERE clause errors
  * Aggregate function errors

Generate the corrected SQL query and IMMEDIATELY execute it using db_query_tool."""),
            ("placeholder", "{messages}")
        ])
        
        return fix_query_prompt | llm.bind_tools([
            tool for tool in tools if tool.name == "db_query_tool"
        ], tool_choice="required")
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Fixing SQL query error")
        messages = state["messages"]
        
        # Hata mesajını ve orijinal query'yi bul
        error_found = False
        for msg in reversed(messages):
            if hasattr(msg, 'content') and "Error:" in str(msg.content):
                error_found = True
                break
        
        if error_found:
            response = self.fix_query_model.invoke({"messages": messages})
            self.log_debug(f"Fix query model response with {len(getattr(response, 'tool_calls', []))} tool calls")
            return {"messages": messages + [response]}
        else:
            response = AIMessage(content="No SQL error found to fix.")
            return {"messages": messages + [response]}


class FinalAnswerNode(BaseNode):
    """Final cevap node'u - SRP"""
    
    def __init__(self):
        super().__init__("FinalAnswerNode")
        self.message_extractor = MessageExtractor()
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Creating final answer")
        messages = state["messages"]
        
        # Kullanıcı sorusunu al
        user_question = self.message_extractor.get_last_user_question(messages)
        
        # Query sonucunu bul
        query_result, executed_query = self.message_extractor.find_tool_result(messages, "call_")
        
        if query_result and not query_result.startswith("Error:") and query_result != "":
            final_text = f"""✅ **Query Başarıyla Çalıştırıldı!**

📋 **Sorunuz:** {user_question}

🔍 **Çalıştırılan SQL:**
```sql
{executed_query}
```

📊 **Sonuçlar:**
{query_result}

💡 **Özet:** Sorgunuz başarıyla çalıştırıldı ve sonuçlar yukarıda görüntülendi."""
        else:
            final_text = f"""❌ **Query Hatası**

📋 **Sorunuz:** {user_question}

Query sonucu bulunamadı veya hata oluştu. Lütfen tekrar deneyin."""
        
        response = AIMessage(content=final_text)
        self.log_debug(f"Final answer created for question: {user_question[:50]}...")
        return {"messages": messages + [response]}
    """Final cevap node'u - SRP"""
    
    def __init__(self):
        super().__init__("FinalAnswerNode")
        self.message_extractor = MessageExtractor()
    
    def execute(self, state: State) -> Dict[str, Any]:
        self.log_debug("Creating final answer")
        messages = state["messages"]
        
        # Kullanıcı sorusunu al
        user_question = self.message_extractor.get_last_user_question(messages)
        
        # Query sonucunu bul
        query_result, executed_query = self.message_extractor.find_tool_result(messages, "call_")
        
        if query_result and not query_result.startswith("Error:") and query_result != "":
            final_text = f"""✅ **Query Başarıyla Çalıştırıldı!**

📋 **Sorunuz:** {user_question}

🔍 **Çalıştırılan SQL:**
```sql
{executed_query}
```

📊 **Sonuçlar:**
{query_result}

💡 **Özet:** Sorgunuz başarıyla çalıştırıldı ve sonuçlar yukarıda görüntülendi."""
        else:
            final_text = f"""❌ **Query Hatası**

📋 **Sorunuz:** {user_question}

Query sonucu bulunamadı veya hata oluştu. Lütfen tekrar deneyin."""
        
        response = AIMessage(content=final_text)
        self.log_debug(f"Final answer created for question: {user_question[:50]}...")
        return {"messages": messages + [response]}


# SOLID Principle 3: Liskov Substitution Principle (LSP)
class RouterInterface(ABC):
    """Router'lar için interface - LSP"""
    
    @abstractmethod
    def route(self, state: State) -> str:
        pass


class SimpleRouter(RouterInterface):
    """Basit routing implementasyonu - LSP"""
    
    def route(self, state: State) -> str:
        messages = state["messages"]
        
        print(f"[DEBUG] Simple routing, messages: {len(messages)}")
        
        # MasterReport isteği mi kontrol et
        if len(messages) >= 1:
            user_question = MessageExtractor.get_last_user_question(messages)
            if TableSelector.is_master_report_request(user_question):
                print("[DEBUG] MasterReport request detected")
                
                # MasterReport setup yapıldı mı kontrol et
                master_report_setup = any(
                    hasattr(msg, 'tool_call_id') and "master_report_setup" in str(msg.tool_call_id) 
                    for msg in messages
                )
                
                if not master_report_setup:
                    print("[DEBUG] → master_report_setup")
                    return "master_report_setup"
        
        # Son 2 mesajı kontrol et
        if len(messages) >= 2:
            last_msg = messages[-1]
            prev_msg = messages[-2] if len(messages) > 1 else None
            
            # Son mesaj ToolMessage ise
            if hasattr(last_msg, 'tool_call_id'):
                tool_call_id = str(last_msg.tool_call_id)
                print(f"[DEBUG] Found tool message with ID: {tool_call_id}")
                
                if "list_tables" in tool_call_id:
                    print("[DEBUG] → get_schema")
                    return "get_schema"
                elif "schema" in tool_call_id:
                    print("[DEBUG] → execute_query")
                    return "execute_query"
                elif "master_report_setup" in tool_call_id:
                    print("[DEBUG] → execute_query (MasterReport)")
                    return "execute_query"
                elif "call_" in tool_call_id:  # OpenAI tool call ID
                    # SQL hatası var mı kontrol et
                    if hasattr(last_msg, 'content') and "Error:" in str(last_msg.content):
                        print("[DEBUG] → fix_query (SQL Error detected)")
                        return "fix_query"
                    else:
                        print("[DEBUG] → final_answer")
                        return "final_answer"
            
            # Önceki mesajda tool call varsa
            if prev_msg and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                for tool_call in prev_msg.tool_calls:
                    if tool_call.get('name') == "db_query_tool":
                        # SQL hatası var mı kontrol et
                        if hasattr(last_msg, 'content') and "Error:" in str(last_msg.content):
                            print("[DEBUG] → fix_query (SQL Error in tool result)")
                            return "fix_query"
                        else:
                            print("[DEBUG] → final_answer (db_query_tool found)")
                            return "final_answer"
        
        print("[DEBUG] → end")
        return "end"


# SOLID Principle 4: Interface Segregation Principle (ISP)
class ToolManagerInterface(ABC):
    """Tool yönetimi için interface - ISP"""
    
    @abstractmethod
    def get_tools(self) -> List:
        pass


class SQLToolManager(ToolManagerInterface):
    """SQL araçları yönetimi - ISP"""
    
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
    
    def get_tools(self) -> List:
        langgraph_tools = LangGraphSQLTools(self.db, self.llm)
        custom_tools = CustomSQLTools(self.db)
        return langgraph_tools.get_basic_tools() + custom_tools.get_custom_tools()


# SOLID Principle 5: Dependency Inversion Principle (DIP)
class NodeFactory:
    """Node'ları oluşturma sorumluluğu - DIP"""
    
    @staticmethod
    def create_nodes(llm, tools) -> Dict[str, NodeInterface]:
        return {
            "list_tables": ListTablesNode(),
            "get_schema": GetSchemaNode(),
            "master_report_setup": MasterReportNode(),
            "execute_query": ExecuteQueryNode(llm, tools),
            "fix_query": FixQueryNode(llm, tools),
            "final_answer": FinalAnswerNode()
        }


@register_graph("text2sql")
class Text2SQLGraph(BaseGraph):
    """SOLID prensiplerine uygun Text2SQL Graph - DIP"""
    
    def __init__(self, llm, db = None):
        super().__init__(llm=llm, state_class=State)

        if db is None:
            db = SQLDatabase.from_uri(config.database.uri)

        self.db = db
        
        # Dependency Injection - DIP
        self.tool_manager = SQLToolManager(db, llm.get_chat())
        self.tools = self.tool_manager.get_tools()
        self.nodes = NodeFactory.create_nodes(llm.get_chat(), self.tools)
        self.router = SimpleRouter()
        self.tool_node = ToolNode(self.tools)
    
    def build_graph(self):
        """Graph'ı oluştur - Temiz ve SOLID"""
        memory = MemorySaver()
        graph = StateGraph(State)
        
        # Node'ları ekle
        for node_name, node_instance in self.nodes.items():
            graph.add_node(f"step_{node_name}", node_instance.execute)
        
        graph.add_node("tools", self.tool_node)
        
        # Edge'leri ekle
        graph.add_edge(START, "step_list_tables")
        graph.add_edge("step_list_tables", "tools")
        graph.add_edge("step_get_schema", "tools")
        graph.add_edge("step_master_report_setup", "tools")  # MasterReport setup edge
        graph.add_edge("step_execute_query", "tools")
        graph.add_edge("step_fix_query", "tools")  # Fix query edge
        
        # Routing
        graph.add_conditional_edges(
            "tools",
            self.router.route,
            {
                "get_schema": "step_get_schema",
                "master_report_setup": "step_master_report_setup",
                "execute_query": "step_execute_query",
                "fix_query": "step_fix_query",  # Bu satır eksikti!
                "final_answer": "step_final_answer",
                "end": END
            }
        )
        
        graph.add_edge("step_final_answer", END)
        
        return graph.compile(checkpointer=memory)