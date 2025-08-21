# src/graphs/generic_sql_graph.py
"""
Generic SQL Graph - Configurable ERP Assistant
Bu graph dışarıdan prompt, SQL, table açıklamaları vs. alarak istediğiniz assistant'ı oluşturur
Master_report_graph.py'den ilham alınmıştır ancak tamamen configurable'dır
"""

import os
import re
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from datetime import datetime
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from pydantic import BaseModel, Field

from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models.models import LLM, State
from src.services.app_logger import log
from src.services.config_loader import ConfigLoader

# ================================
# CONFIGURATION MODEL
# ================================

class SQLAgentConfiguration(BaseModel):
    """Configurable parameters for Generic SQL Agent"""
    
    # Core Assistant Settings
    system_prompt: str = Field(
        default="Sen bir ERP uzmanı ve SQL sorgulama asistanısın. Görevin verilen bilgileri kullanarak senden istenilen sorguyu ve bilgiyi vermen.",
        description="Ana sistem promptu - assistant'ın rolünü ve davranışını belirler"
    )
    
    assistant_name: str = Field(
        default="ERP SQL Assistant",
        description="Assistant'ın adı"
    )
    
    # Database Configuration
    initial_setup_sql: str = Field(
        default="SELECT 1 as test_connection",
        description="Graph başlamadan önce çalıştırılacak setup SQL (temp table oluşturma vs.)"
    )

    table_name: str = Field(
        default="",
        description="Sorgu oluşturulacak ve çekilecek ana tablo adı"
    )

    table_columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Tablo kolonları bilgileri [{'name': 'kolon_adi', 'type': 'veri_tipi', 'description': 'açıklama'}]"
    )
    
    table_description: str = Field(
        default="Genel ERP tabloları",
        description="Kullanılacak tablonun açıklaması ve yapısı gibi ek detaylar"
    )
    
    cleanup_sql: str = Field(
        default="-- No cleanup needed",
        description="Graph sonunda çalıştırılacak temizlik SQL (temp table silme vs.)"
    )
    
    # Response Configuration
    response_template: str = Field(
        default="""Sen yardımcı bir müşteri hizmetleri asistanısın.

SQL sonucunu kullanıcı dostu bir şekilde açıkla.

Kullanıcı Sorusu: {question}
SQL Sorgusu: {query}  
SQL Sonucu: {result}

Kurallar:
- Emoji kullan (📊, 💰, 📈, ✅, 🏪)
- Sayıları formatla
- Önemli bilgileri vurgula
- Kısa ve net ol
- Türkçe yanıt ver

Formatlanmış yanıt:""",
        description="Final yanıt template'i"
    )
    
    # Security and Performance Settings
    max_result_rows: int = Field(
        default=100,
        description="Maximum döndürülecek satır sayısı"
    )
    
    allowed_operations: List[str] = Field(
        default=["SELECT"],
        description="İzin verilen SQL operasyonları"
    )
    
    enable_query_fixing: bool = Field(
        default=True,
        description="SQL hata düzeltme özelliğini etkinleştir"
    )
    
    # Language Model Settings
    model: Annotated[
        Literal[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="gpt-4o-mini",
        description="Kullanılacak language model"
    )
    
    temperature: float = Field(
        default=0.0,
        description="Model temperature ayarı"
    )
    
    # Data Exploration Settings
    max_distinct_values: int = Field(
        default=20,
        description="Her kolon için maksimum distinct değer sayısı"
    )

# ================================
# STATE DEFINITION
# ================================

class GenericSQLState(TypedDict):
    """Generic SQL Assistant State"""
    # Core conversation
    question: str                              # User question
    messages: List[Any]                        # Chat messages
    
    # Configuration data
    table_name: str                           # Main table name
    table_columns: List[Dict[str, str]]       # Table column definitions
    table_description: str                    # Table description
    
    # Data exploration
    column_distinct_values: Dict[str, List]   # Distinct values per column
    data_exploration_completed: bool          # Data exploration status
    
    # SQL workflow  
    setup_completed: bool                     # Setup SQL executed?
    query: str                               # Generated SQL
    result: str                              # SQL execution result
    answer: str                              # Final formatted answer
    error: Optional[str]                     # Error message
    fixed_query: Optional[str]               # Fixed SQL query
    cleanup_completed: bool                  # Cleanup executed?
    
    # Metadata
    metadata: Dict[str, Any]                 # Additional metadata

# ================================
# CORE FUNCTIONS
# ================================

def initialize_state(state: GenericSQLState, config: RunnableConfig) -> GenericSQLState:
    """Initialize state with configuration data"""
    logger = log.get(module="generic_sql", function="initialize_state")
    
    try:
        # Get configuration
        configurable = config.get("configurable", {})
        
        # Extract user question from messages if not already set
        if not state.get("question") and state.get("messages"):
            messages = state["messages"]
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "human":
                    state["question"] = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    state["question"] = msg.get("content", "")
                    break
        
        # Initialize from config
        state["table_name"] = configurable.get("table_name", "")
        state["table_columns"] = configurable.get("table_columns", [])
        state["table_description"] = configurable.get("table_description", "")
        
        # Initialize data exploration state
        state["column_distinct_values"] = {}
        state["data_exploration_completed"] = False
        
        # Initialize workflow state
        state["setup_completed"] = False
        state["query"] = ""
        state["result"] = ""
        state["answer"] = ""
        state["error"] = None
        state["fixed_query"] = None
        state["cleanup_completed"] = False
        state["metadata"] = {"initialization_time": datetime.now().isoformat()}
        
        logger.info("State initialized successfully", 
                   table_name=state["table_name"],
                   columns_count=len(state["table_columns"]))
        
    except Exception as e:
        logger.error("State initialization failed", error=str(e))
        state["error"] = f"Initialization error: {str(e)}"
    
    return state


def setup_database(state: GenericSQLState, config: RunnableConfig, db: SQLDatabase) -> GenericSQLState:
    """Execute initial setup SQL"""
    logger = log.get(module="generic_sql", function="setup_database")
    
    try:
        # Get configuration
        configurable = config.get("configurable", {})
        setup_sql = configurable.get("initial_setup_sql", "SELECT 1 as test_connection")
        
        if setup_sql and setup_sql.strip() != "SELECT 1 as test_connection":
            # Execute setup SQL
            execute_tool = QuerySQLDatabaseTool(db=db)
            result = execute_tool.invoke(setup_sql)
            
            logger.info("Database setup completed", 
                       sql_length=len(setup_sql),
                       result_preview=str(result)[:100])
        
        state["setup_completed"] = True
        state["metadata"] = {"setup_time": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        state["error"] = f"Setup error: {str(e)}"
        state["setup_completed"] = False
    
    return state


def explore_column_data(state: GenericSQLState, config: RunnableConfig, db: SQLDatabase) -> GenericSQLState:
    """Collect distinct values from each column for better SQL generation"""
    logger = log.get(module="generic_sql", function="explore_column_data")
    
    try:
        table_name = state.get("table_name", "")
        table_columns = state.get("table_columns", [])
        
        if not table_name or not table_columns:
            logger.warning("No table name or columns provided, skipping data exploration")
            state["data_exploration_completed"] = False
            return state
        
        execute_tool = QuerySQLDatabaseTool(db=db)
        column_distinct_values = {}
        
        # Get configurable limits
        configurable = config.get("configurable", {})
        max_distinct_values = configurable.get("max_distinct_values", 20)
        
        for column in table_columns:
            column_name = column.get("name", "")
            column_type = column.get("type", "").lower()
            
            if not column_name:
                continue
            
            try:
                # Skip certain data types that might have too many distinct values
                if any(skip_type in column_type for skip_type in ['text', 'ntext', 'image', 'binary']):
                    logger.info(f"Skipping column {column_name} due to data type: {column_type}")
                    continue
                
                # Create distinct query with limit
                distinct_query = f"""
                SELECT DISTINCT TOP {max_distinct_values} [{column_name}]
                FROM [{table_name}]
                WHERE [{column_name}] IS NOT NULL
                ORDER BY [{column_name}]
                """
                
                result = execute_tool.invoke(distinct_query)
                
                if result:
                    # Parse result into list of values
                    result_str = str(result).strip()
                    
                    # Extract values from result string (basic parsing)
                    values = []
                    if result_str and result_str != "[]":
                        # Split by newlines and clean up
                        lines = result_str.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('(') and line != '[]':
                                # Remove parentheses and quotes
                                clean_value = line.strip("(),'\"")
                                if clean_value and clean_value not in ['None', 'NULL']:
                                    values.append(clean_value)
                    
                    # Limit values to prevent overwhelming the prompt
                    if len(values) > max_distinct_values:
                        values = values[:max_distinct_values]
                    
                    column_distinct_values[column_name] = values
                    
                    logger.info(f"Collected {len(values)} distinct values for column {column_name}")
                
            except Exception as col_error:
                logger.warning(f"Failed to get distinct values for column {column_name}", 
                             error=str(col_error))
                column_distinct_values[column_name] = []
        
        state["column_distinct_values"] = column_distinct_values
        state["data_exploration_completed"] = True
        
        # Update metadata
        state["metadata"]["data_exploration_time"] = datetime.now().isoformat()
        state["metadata"]["explored_columns_count"] = len(column_distinct_values)
        
        logger.info("Data exploration completed successfully", 
                   columns_explored=len(column_distinct_values),
                   total_distinct_values=sum(len(vals) for vals in column_distinct_values.values()))
    
    except Exception as e:
        logger.error("Data exploration failed", error=str(e))
        state["column_distinct_values"] = {}
        state["data_exploration_completed"] = False
        # Don't set error as this is not critical for the workflow
    
    return state


def write_query(state: GenericSQLState, config: RunnableConfig, llm: LLM) -> GenericSQLState:
    """Generate SQL query using configurable system prompt"""
    logger = log.get(module="generic_sql", function="write_query")
    
    try:
        # Get configuration
        configurable = config.get("configurable", {})
        system_prompt = configurable.get("system_prompt", "Sen bir SQL uzmanısın.")
        max_rows = configurable.get("max_result_rows", 100)
        
        # Get data from state
        table_name = state.get("table_name", "")
        table_columns = state.get("table_columns", [])
        table_description = state.get("table_description", "")
        column_distinct_values = state.get("column_distinct_values", {})
        
        # Build enhanced table info section with distinct values
        table_info = ""
        if table_name:
            table_info = f"\n## ANA TABLO: {table_name}\n"
            
            if table_columns:
                table_info += "\n### KOLONLAR VE MEVCUT DEĞERLER:\n"
                for col in table_columns:
                    name = col.get("name", "")
                    col_type = col.get("type", "")
                    desc = col.get("description", "")
                    
                    table_info += f"- **{name}** ({col_type}): {desc}\n"
                    
                    # Add distinct values if available
                    if name in column_distinct_values and column_distinct_values[name]:
                        values = column_distinct_values[name]
                        if len(values) <= 10:
                            values_str = ", ".join([f"'{v}'" for v in values])
                            table_info += f"  📋 Mevcut değerler: {values_str}\n"
                        else:
                            sample_values = values[:8]
                            values_str = ", ".join([f"'{v}'" for v in sample_values])
                            table_info += f"  📋 Örnek değerler: {values_str} (ve {len(values)-8} tane daha)\n"
                    table_info += "\n"
            
            if table_description and table_description != "Genel ERP tabloları":
                table_info += f"### AÇIKLAMA:\n{table_description}\n"
        
        # Create enhanced system prompt with table info
        enhanced_prompt = f"""{system_prompt}
{table_info}
## SQL KURALLARI:
- Her zaman TOP {max_rows} kullan (LIMIT yerine)
- Sadece SELECT sorguları kullan
- MSSQL syntax'ını kullan
- Güvenli ve optimize edilmiş sorgular yaz
- Ana tablo: {table_name if table_name else 'Belirtilmemiş'}

Kullanıcı Sorusu: {{question}}

Bu soruya uygun syntactically correct MSSQL sorgusu üret:"""

        # Create structured output for SQL
        class QueryOutput(BaseModel):
            query: Annotated[str, ..., "Valid MSSQL query"]
        
        # Create prompt and generate SQL
        prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_prompt),
            ("human", "{question}")
        ])
        
        structured_llm = llm.get_chat().with_structured_output(QueryOutput)
        
        message = prompt.invoke({"question": state["question"]})
        result = structured_llm.invoke(message)
        
        # Clean SQL
        sql = result["query"]
        sql = clean_sql(sql)
        
        # Security check
        allowed_ops = configurable.get("allowed_operations", ["SELECT"])
        if not is_safe_query(sql, allowed_ops):
            raise ValueError("Unsafe SQL query detected")
        
        state["query"] = sql
        logger.info("SQL query generated successfully", 
                   sql_preview=sql[:100])
        
    except Exception as e:
        logger.error("Query generation failed", error=str(e))
        state["error"] = str(e)
        state["query"] = generate_safe_fallback_query(configurable.get("max_result_rows", 100))
    
    return state


def execute_query(state: GenericSQLState, config: RunnableConfig, db: SQLDatabase) -> GenericSQLState:
    """Execute generated SQL query"""
    logger = log.get(module="generic_sql", function="execute_query")
    
    try:
        if state.get("error"):
            # Skip execution if there are errors
            state["result"] = f"Query not executed due to error: {state['error']}"
            return state
        
        # Execute query
        execute_tool = QuerySQLDatabaseTool(db=db)
        result = execute_tool.invoke(state["query"])
        
        if result:
            state["result"] = str(result)
            logger.info("Query executed successfully", 
                       result_length=len(str(result)))
        else:
            state["result"] = "Sonuç bulunamadı"
            logger.warning("Query returned empty result")
    
    except Exception as e:
        logger.error("Query execution failed", 
                    error=str(e),
                    query=state.get("query", ""))
        
        state["error"] = str(e)
        state["result"] = f"EXECUTION ERROR: {str(e)}"
    
    return state


def fix_query(state: GenericSQLState, config: RunnableConfig, llm: LLM, db: SQLDatabase) -> GenericSQLState:
    """Fix SQL query if there were execution errors"""
    logger = log.get(module="generic_sql", function="fix_query")
    
    # Check if fixing is enabled and needed
    configurable = config.get("configurable", {})
    fix_enabled = configurable.get("enable_query_fixing", True)
    
    if not fix_enabled or not state.get("error"):
        return state
    
    try:
        table_name = configurable.get("table_name", "")
        table_columns = configurable.get("table_columns", [])
        table_description = configurable.get("table_description", "")
        
        # Build table info for fixing
        table_info = ""
        if table_name:
            table_info = f"## ANA TABLO: {table_name}\n"
            
            if table_columns:
                table_info += "\n### KOLONLAR:\n"
                for col in table_columns:
                    name = col.get("name", "")
                    col_type = col.get("type", "")
                    desc = col.get("description", "")
                    table_info += f"- {name} ({col_type}): {desc}\n"
            
            if table_description:
                table_info += f"\n### AÇIKLAMA:\n{table_description}\n"
        
        # Create fixing prompt
        fix_prompt = f"""Sen bir MSSQL uzmanısın. SQL hatasını düzelt.

{table_info}

## HATA BİLGİSİ:
Orjinal SQL: {state.get('query', '')}
Hata Mesajı: {state.get('error', '')}
Kullanıcı Sorusu: {state.get('question', '')}

## GÖREV:
Hatayı analiz et ve düzeltilmiş MSSQL sorgusu oluştur.
Sadece düzeltilmiş SQL'i döndür, açıklama yapma.

Düzeltilmiş SQL:"""

        response = llm.get_chat().invoke([
            SystemMessage(content=fix_prompt)
        ])
        
        fixed_sql = clean_sql(response.content)
        
        # Try executing fixed query
        execute_tool = QuerySQLDatabaseTool(db=db)
        result = execute_tool.invoke(fixed_sql)
        
        # If successful, update state
        state["fixed_query"] = fixed_sql
        state["query"] = fixed_sql  # Use fixed query
        state["result"] = str(result) if result else "Düzeltilmiş sorgu boş sonuç döndürdü"
        state["error"] = None  # Clear error
        
        logger.info("Query fixed and executed successfully")
        
    except Exception as e:
        logger.error("Query fixing failed", error=str(e))
        # Keep original error and result
        state["fixed_query"] = f"-- Fix failed: {str(e)}"
    
    return state


def generate_answer(state: GenericSQLState, config: RunnableConfig, llm: LLM) -> GenericSQLState:
    """Generate final user-friendly answer"""
    logger = log.get(module="generic_sql", function="generate_answer")
    
    try:
        # Get configuration
        configurable = config.get("configurable", {})
        response_template = configurable.get("response_template", DEFAULT_RESPONSE_TEMPLATE)
        assistant_name = configurable.get("assistant_name", "SQL Assistant")
        
        # Handle error cases
        if state.get("error") and not state.get("fixed_query"):
            state["answer"] = format_error_message(
                state["error"], 
                state["question"], 
                assistant_name
            )
            return state
        
        # Create answer prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", response_template),
            ("human", "Yanıtı oluştur")
        ])
        
        # Generate answer
        message = prompt.invoke({
            "question": state["question"],
            "query": state.get("query", ""),
            "result": state.get("result", "")
        })
        
        response = llm.get_chat().invoke(message)
        answer = response.content
        
        # Add metadata footer
        answer = add_metadata_footer(answer, state, assistant_name)
        
        state["answer"] = answer
        logger.info("Final answer generated", answer_length=len(answer))
        
    except Exception as e:
        logger.error("Answer generation failed", error=str(e))
        state["answer"] = f"Yanıt oluşturulamadı: {str(e)}"
    
    return state


def cleanup_database(state: GenericSQLState, config: RunnableConfig, db: SQLDatabase) -> GenericSQLState:
    """Execute cleanup SQL and ensure messages compatibility"""
    logger = log.get(module="generic_sql", function="cleanup_database")
    
    try:
        # Get configuration
        configurable = config.get("configurable", {})
        cleanup_sql = configurable.get("cleanup_sql", "-- No cleanup needed")
        
        if cleanup_sql and not cleanup_sql.strip().startswith("-- No cleanup"):
            # Execute cleanup
            execute_tool = QuerySQLDatabaseTool(db=db)
            result = execute_tool.invoke(cleanup_sql)
            
            logger.info("Database cleanup completed", 
                       sql_length=len(cleanup_sql))
        
        state["cleanup_completed"] = True
        
        # Update metadata
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["cleanup_time"] = datetime.now().isoformat()
        
        # CRITICAL FIX: Ensure messages key exists for supervisor compatibility
        if "messages" not in state:
            state["messages"] = []
        
        # Add AI message with the final answer
        answer = state.get("answer", "Sonuç bulunamadı.")
        if answer:
            from langchain.schema import AIMessage
            ai_message = AIMessage(content=answer)
            state["messages"].append(ai_message)
        
    except Exception as e:
        logger.error("Database cleanup failed", error=str(e))
        state["cleanup_completed"] = False
        # Don't fail the whole process for cleanup errors
        
        # Even on error, ensure messages exists
        if "messages" not in state:
            from langchain.schema import AIMessage
            error_msg = f"Veritabanı işlemi sırasında hata oluştu: {str(e)}"
            state["messages"] = [AIMessage(content=error_msg)]
    
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
    
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    
    return sql.strip()


def is_safe_query(sql: str, allowed_operations: List[str]) -> bool:
    """Check if SQL query is safe"""
    sql_upper = sql.upper()
    
    # Check if starts with allowed operation
    starts_with_allowed = any(
        sql_upper.strip().startswith(op.upper()) 
        for op in allowed_operations
    )
    
    if not starts_with_allowed:
        return False
    
    # Check for dangerous operations
    dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE', 'DECLARE']
    
    for op in dangerous:
        if op not in [a.upper() for a in allowed_operations] and op in sql_upper:
            return False
    
    return True


def generate_safe_fallback_query(max_rows: int = 100) -> str:
    """Generate a safe fallback query"""
    return f"SELECT TOP {max_rows} 'Güvenli sorgu oluşturulamadı' as Mesaj"


def format_error_message(error: str, question: str, assistant_name: str) -> str:
    """Format error message for user"""
    return f"""
❌ **{assistant_name} - İşlem Başarısız**

**Sorunuz:** {question}
**Hata:** {error}

🔄 **Öneriler:**
• Sorunuzu daha basit ifade edin
• Tablo ve sütun adlarını kontrol edin
• Tarih formatlarını doğru kullanın

📞 Yardım için destek ekibi ile iletişime geçebilirsiniz.
"""


def add_metadata_footer(answer: str, state: GenericSQLState, assistant_name: str) -> str:
    """Add metadata footer to answer"""
    current_time = datetime.now().strftime("%H:%M")
    
    metadata_parts = [
        f"📌 **{assistant_name} Bilgi**",
        f"• Zaman: {current_time}"
    ]
    
    if state.get("fixed_query"):
        metadata_parts.append("• Sorgu düzeltildi")
    
    if state.get("metadata", {}).get("setup_time"):
        metadata_parts.append("• Veritabanı hazırlandı")
    
    footer = "\n\n---\n" + "\n".join(metadata_parts)
    
    return answer + footer


# Default response template
DEFAULT_RESPONSE_TEMPLATE = """Sen yardımcı bir müşteri hizmetleri asistanısın.

SQL sonucunu kullanıcı dostu bir şekilde açıkla.

Kullanıcı Sorusu: {question}
SQL Sorgusu: {query}
SQL Sonucu: {result}

Kurallar:
- Emoji kullan (📊, 💰, 📈, ✅, 🏪)
- Sayıları formatla
- Önemli bilgileri vurgula
- Kısa ve net ol
- Türkçe yanıt ver

Formatlanmış yanıt:"""


# ================================
# CONDITIONAL ROUTING
# ================================

def should_fix_query(state: GenericSQLState) -> str:
    """Decide whether to fix query or generate answer"""
    if state.get("error") and not state.get("fixed_query"):
        return "fix_query"
    else:
        return "generate_answer"


# ================================
# MAIN GRAPH CLASS
# ================================

@register_graph("generic_sql")
class GenericSQLGraph(BaseGraph):
    """Generic and Configurable SQL Graph for ERP Assistants"""
    
    def __init__(self, llm: LLM, db: SQLDatabase = None, memory: MemorySaver = None):
        super().__init__(llm=llm, state_class=State, memory=memory)
        self.logger = log.get(module="generic_sql_graph")
        
        # Database setup
        if db is None:
            config = ConfigLoader.load_config("config/text2sql_config.yaml")
            db = SQLDatabase.from_uri(config.database.uri)
        
        self.db = db
        self.logger.info("Generic SQL Graph initialized")
    
    async def make_graph(self, config: RunnableConfig):
        """Async factory method for creating configured graph"""
        return self.build_graph()
    
    def build_graph(self):
        """Build the configurable SQL graph using React Agent"""
        
        self.logger.info("Building configurable SQL graph")
        
        try:
            # Create memory if not provided
            if self.memory is None:
                self.memory = MemorySaver()
            
            # Create state graph with configuration schema
            graph_builder = StateGraph(GenericSQLState, config_schema=SQLAgentConfiguration)
            
            # Node wrapper functions with config injection
            def initialize_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return initialize_state(state, config)
            
            def setup_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return setup_database(state, config, self.db)
            
            def explore_data_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return explore_column_data(state, config, self.db)
            
            def write_query_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return write_query(state, config, self.llm)
            
            def execute_query_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return execute_query(state, config, self.db)
            
            def fix_query_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return fix_query(state, config, self.llm, self.db)
            
            def generate_answer_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return generate_answer(state, config, self.llm)
            
            def cleanup_node(state: GenericSQLState, *, config: RunnableConfig) -> GenericSQLState:
                return cleanup_database(state, config, self.db)
            
            # Add nodes to graph
            graph_builder.add_node("initialize", initialize_node)
            graph_builder.add_node("setup", setup_node)
            graph_builder.add_node("explore_data", explore_data_node)
            graph_builder.add_node("write_query", write_query_node)
            graph_builder.add_node("execute_query", execute_query_node)
            graph_builder.add_node("fix_query", fix_query_node)
            graph_builder.add_node("generate_answer", generate_answer_node)
            graph_builder.add_node("cleanup", cleanup_node)
            
            # Define enhanced graph flow
            graph_builder.add_edge(START, "initialize")
            graph_builder.add_edge("initialize", "setup")
            graph_builder.add_edge("setup", "explore_data")
            graph_builder.add_edge("explore_data", "write_query")
            graph_builder.add_edge("write_query", "execute_query")
            
            # Conditional routing after execution
            graph_builder.add_conditional_edges(
                "execute_query",
                should_fix_query,
                {
                    "fix_query": "fix_query",
                    "generate_answer": "generate_answer"
                }
            )
            
            graph_builder.add_edge("fix_query", "generate_answer")
            graph_builder.add_edge("generate_answer", "cleanup")
            graph_builder.add_edge("cleanup", END)
            
            # Compile with checkpointer
            compiled_graph = graph_builder.compile(
                checkpointer=self.memory,
                name="generic_sql_graph"
            )
            
            self.logger.info("Generic SQL graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            self.logger.error("Failed to build graph", error=str(e))
            raise
    
    def invoke(self, messages: List[Any], config: Optional[Dict] = None) -> str:
        """Main entry point - compatible with existing chatbot interface"""
        try:
            # Extract question from messages
            question = self._extract_question(messages)
            
            if not question:
                return self._create_help_message()
            
            # Create initial state
            initial_state = GenericSQLState(
                question=question,
                setup_completed=False,
                query="",
                result="",
                answer="",
                error=None,
                fixed_query=None,
                cleanup_completed=False,
                metadata={}
            )
            
            # Build graph if needed
            if not hasattr(self, '_compiled_graph'):
                self._compiled_graph = self.build_graph()
            
            # Default config
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
            
            # Run the graph
            result = self._compiled_graph.invoke(initial_state, config)
            
            # Return the answer
            return result.get("answer", "Yanıt oluşturulamadı")
            
        except Exception as e:
            self.logger.error("Processing failed", error=str(e))
            return self._create_error_message(str(e))
    
    def _extract_question(self, messages: List[Any]) -> str:
        """Extract question from various message formats"""
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
🤖 **Generic SQL Assistant**

Size nasıl yardımcı olabilirim?

Bu assistant tamamen konfigüre edilebilir ve çeşitli SQL tabloları ile çalışabilir.

📝 Sorularınızı yazın ve size SQL tabanlı yanıtlar sunayım.

💡 **İpucu:** Spesifik veriler, raporlar, analizler hakkında sorular sorabilirsiniz.
"""
    
    def _create_error_message(self, error: str) -> str:
        """Create error message"""
        return f"""
❌ **Sistem Hatası**

Üzgünüm, bir hata oluştu:
{error}

🔄 Lütfen tekrar deneyin veya sorunuzu basitleştirin.
"""


# ================================
# EXAMPLE CONFIGURATIONS
# ================================

def get_master_report_config():
    """Master Report Document için örnek konfigürasyon"""
    return {
        "system_prompt": """Sen bir ERP uzmanı ve SQL sorgulama asistanısın.
Restoran/kafe işletmesi için MasterReportDocument tablosundan veri çekiyorsun.""",
        
        "assistant_name": "Master Report Assistant",
        
        "table_description": """
## TABLO YAPISI - MasterReportDocument

### Ana Belge Alanları:
- DocumentId: Belge/Adisyon benzersiz ID'si
- DocumentDate: İşlem tarihi ve saati
- DocumentStatus: Belge durumu
- DocumentTypeName: Belge tipi adı
- BranchName: Şube adı
- ProductName: Ürün adı
- CategoryName: Kategori adı
- RowAmount: Satır tutarı
- DiscountAmountTotal: İndirim tutarı
""",
        
        "initial_setup_sql": "SELECT 1 as connection_test",
        "cleanup_sql": "-- No cleanup needed for MasterReportDocument",
        "max_result_rows": 100,
        "enable_query_fixing": True
    }


def get_temp_table_config():
    """Temp table kullanan assistant için örnek konfigürasyon"""
    return {
        "system_prompt": "Sen bir veri analizi uzmanısın. Geçici tablolar üzerinden analiz yapıyorsun.",
        
        "assistant_name": "Data Analysis Assistant",
        
        "initial_setup_sql": """
        CREATE TABLE #TempSales (
            ProductId INT,
            ProductName NVARCHAR(100),
            Quantity INT,
            Amount DECIMAL(10,2),
            SaleDate DATE
        )
        
        INSERT INTO #TempSales VALUES 
        (1, 'Ürün A', 10, 100.00, '2024-01-01'),
        (2, 'Ürün B', 5, 50.00, '2024-01-01')
        """,
        
        "table_description": """
## TEMP TABLE - #TempSales
- ProductId: Ürün ID
- ProductName: Ürün adı  
- Quantity: Miktar
- Amount: Tutar
- SaleDate: Satış tarihi
        """,
        
        "cleanup_sql": "DROP TABLE IF EXISTS #TempSales",
        
        "response_template": """Veri analizi sonuçlarını açıkla:

Soru: {question}
SQL: {query}
Sonuç: {result}

Analiz sonuçlarını grafiksel açıklamalarla sun."""
    }


# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    """Example usage with different configurations"""
    
    from src.models.models import LLM
    
    # Initialize LLM
    llm = LLM(model="gpt-4o-mini", temperature=0.0)
    
    # Create graph
    graph = GenericSQLGraph(llm)
    compiled = graph.build_graph()
    
    # Example 1: Master Report Configuration
    master_config = {
        "configurable": {
            **get_master_report_config(),
            "thread_id": "master_report_session"
        }
    }
    
    # Example 2: Temp Table Configuration  
    temp_config = {
        "configurable": {
            **get_temp_table_config(),
            "thread_id": "temp_table_session"
        }
    }
    
    print("🚀 Generic SQL Graph configured and ready to use!")
    print("📋 Available configurations: Master Report, Temp Table")
    print("🔧 Fully configurable for any SQL-based assistant needs!")