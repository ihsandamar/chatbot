# src/models/erp_langgraph_models.py
"""
ERP Chatbot için LangGraph uyumlu tip-güvenli modeller
Dedicated TypedDict'ler ile sub-graph state management
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any, Union
from langgraph.graph.message import add_messages, AnyMessage
from enum import Enum
from datetime import datetime

# ==============================================================================
# ENUM DEFINITIONS
# ==============================================================================

class ModuleType(str, Enum):
    """Ana modül tipleri"""
    REPORTING = "reporting"
    SUPPORT = "support"
    DOCUMENTS_TRAINING = "documents_training"
    REQUEST = "request"
    COMPANY_INFO = "company_info"
    OTHER = "other"

class ReportingSubModule(str, Enum):
    """Raporlama alt modülleri"""
    MASTER_REPORT = "master_report"
    ACCOUNTING = "accounting"
    DYNAMIC_REPORTING = "dynamic_reporting"
    SELECT = "select"

class SupportSubModule(str, Enum):
    """Destek alt modülleri"""
    PROGRAM_INSTALLATION = "program_installation"
    PAGE_ROUTING = "page_routing"
    COMMON_ISSUES = "common_issues"
    ERROR_REPORTING = "error_reporting"

class VisualizationType(str, Enum):
    """Görselleştirme tipleri"""
    TABLE = "table"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    AREA_CHART = "area_chart"
    SCATTER_PLOT = "scatter_plot"

class FileExportType(str, Enum):
    """Dosya export tipleri"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"

class RequestStatus(str, Enum):
    """İstek durumları"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PARAMETER_COLLECTION = "parameter_collection"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_VISUALIZATION = "requires_visualization"

# ==============================================================================
# DEDICATED DATA MODELS (TYPE-SAFE)
# ==============================================================================

class UserInfo(TypedDict):
    """Kullanıcı bilgileri için dedicated type"""
    user_id: str
    username: str
    company_id: Optional[str]
    branch_id: Optional[str]
    permissions: List[str]
    session_id: str

class BranchInfo(TypedDict):
    """Şube bilgileri için dedicated type"""
    branch_id: str
    branch_name: str
    city: str
    district: str
    is_active: bool

class DateRangeInfo(TypedDict):
    """Tarih aralığı bilgileri için dedicated type"""
    start_date: str
    end_date: str
    period_type: str  # "daily", "monthly", "yearly", "custom"

class ReportParameters(TypedDict):
    """Rapor parametreleri için dedicated type"""
    report_type: str  # "ciro", "satış", "ürün", "kategori"
    branch_filter: Optional[List[str]]
    date_range: Optional[DateRangeInfo]
    product_filter: Optional[List[str]]
    category_filter: Optional[List[str]]
    customer_filter: Optional[List[str]]
    additional_filters: Optional[Dict[str, Any]]

class SQLQueryInfo(TypedDict):
    """SQL sorgu bilgileri için dedicated type"""
    query: str
    is_valid: bool
    is_select_only: bool
    affected_tables: List[str]
    estimated_rows: Optional[int]
    execution_time: Optional[float]

class ReportResult(TypedDict):
    """Rapor sonucu için dedicated type"""
    data: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    summary_stats: Optional[Dict[str, Any]]
    sql_query: str
    execution_time: float
    generated_at: str

class VisualizationConfig(TypedDict):
    """Görselleştirme konfigürasyonu için dedicated type"""
    chart_type: VisualizationType
    title: str
    x_axis: str
    y_axis: str
    color_scheme: Optional[str]
    show_legend: bool
    chart_options: Optional[Dict[str, Any]]

class VisualizationResult(TypedDict):
    """Görselleştirme sonucu için dedicated type"""
    chart_data: Dict[str, Any]
    chart_config: VisualizationConfig
    chart_url: Optional[str]
    export_options: List[FileExportType]

class ErrorInfo(TypedDict):
    """Hata bilgileri için dedicated type"""
    error_code: str
    error_message: str
    error_type: str  # "sql_error", "parameter_error", "permission_error"
    suggested_solution: Optional[str]
    retry_possible: bool

class SupportTicket(TypedDict):
    """Destek talebi için dedicated type"""
    ticket_id: str
    user_id: str
    issue_type: SupportSubModule
    description: str
    priority: str  # "low", "medium", "high", "critical"
    status: str
    created_at: str
    assigned_to: Optional[str]

class CompanySettings(TypedDict):
    """Firma ayarları için dedicated type"""
    company_id: str
    company_name: str
    erp_modules: List[str]
    active_branches: List[BranchInfo]
    reporting_permissions: Dict[str, List[str]]
    custom_settings: Dict[str, Any]

# ==============================================================================
# SUB GRAPH STATE'LERİ
# ==============================================================================

class SupervisorState(TypedDict):
    """Supervisor Graph'ın state'i - Yönlendirme ve genel koordinasyon"""
    messages: Annotated[List[AnyMessage], add_messages]
    user_info: Optional[UserInfo]
    selected_module: Optional[ModuleType]
    routing_context: Optional[Dict[str, Any]]
    session_data: Optional[Dict[str, Any]]

class ReportingSubState(TypedDict):
    """Reporting Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    selected_sub_module: Optional[ReportingSubModule]
    report_parameters: Optional[ReportParameters]
    parameter_validation_errors: Optional[List[str]]
    missing_parameters: Optional[List[str]]
    is_parameter_collection_complete: bool

class Text2SQLSubState(TypedDict):
    """Text2SQL Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    user_question: str
    sql_query: Optional[SQLQueryInfo]
    table_schema_info: Optional[Dict[str, Any]]
    query_validation_result: Optional[Dict[str, Any]]
    report_result: Optional[ReportResult]
    execution_errors: Optional[List[ErrorInfo]]

class VisualizationSubState(TypedDict):
    """Visualization Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    source_data: Optional[ReportResult]
    visualization_request: Optional[str]
    chart_config: Optional[VisualizationConfig]
    visualization_result: Optional[VisualizationResult]
    export_request: Optional[FileExportType]

class SupportSubState(TypedDict):
    """Support Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    selected_sub_module: Optional[SupportSubModule]
    support_ticket: Optional[SupportTicket]
    issue_analysis: Optional[Dict[str, Any]]
    solution_steps: Optional[List[str]]
    escalation_required: bool

class RequestSubState(TypedDict):
    """Request Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    request_type: Optional[str]
    request_details: Optional[Dict[str, Any]]
    priority_assessment: Optional[str]
    approval_required: bool
    request_status: RequestStatus

class CompanyInfoSubState(TypedDict):
    """Company Info Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    info_request_type: Optional[str]
    company_settings: Optional[CompanySettings]
    requested_information: Optional[Dict[str, Any]]

class OtherSubState(TypedDict):
    """Other/General Sub Graph'ın internal state'i"""
    messages: Annotated[List[AnyMessage], add_messages]
    external_request_type: Optional[str]  # "web_search", "currency", "weather"
    search_query: Optional[str]
    external_api_result: Optional[Dict[str, Any]]
    fallback_suggestions: Optional[List[str]]

# ==============================================================================
# ANA ERP CHATBOT STATE (TYPE-SAFE)
# ==============================================================================

class ERPChatbotState(TypedDict):
    """Ana ERP Chatbot state'i - Tüm sub-graph'lar bu state'i güncelleyebilir"""
    
    # Temel mesajlaşma
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Kullanıcı ve oturum bilgileri
    user_info: Optional[UserInfo]
    session_context: Optional[Dict[str, Any]]
    
    # Aktif modül ve durum tracking
    current_module: Optional[ModuleType]
    current_sub_module: Optional[str]
    workflow_step: str
    
    # Raporlama sonuçları (Reporting & Text2SQL sub-graphs güncelleyecek)
    report_parameters: Optional[ReportParameters]
    report_result: Optional[ReportResult]
    sql_query_info: Optional[SQLQueryInfo]
    
    # Görselleştirme sonuçları (Visualization sub-graph güncelleyecek)
    visualization_config: Optional[VisualizationConfig]
    visualization_result: Optional[VisualizationResult]
    
    # Destek sonuçları (Support sub-graph güncelleyecek)
    support_ticket: Optional[SupportTicket]
    support_resolution: Optional[Dict[str, Any]]
    
    # Talep sonuçları (Request sub-graph güncelleyecek)
    request_info: Optional[Dict[str, Any]]
    request_status: Optional[RequestStatus]
    
    # Firma bilgileri (Company Info sub-graph güncelleyecek)
    company_settings: Optional[CompanySettings]
    
    # Genel/Diğer sonuçları (Other sub-graph güncelleyecek)
    external_data: Optional[Dict[str, Any]]
    
    # Hata yönetimi ve durum kontrolü
    errors: Optional[List[ErrorInfo]]
    requires_user_input: bool
    pending_parameters: Optional[List[str]]
    
    # Metadata ve tracking
    conversation_metadata: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]

# ==============================================================================
# WORKFLOW COORDINATION MODELS
# ==============================================================================

class WorkflowTransition(TypedDict):
    """İş akışı geçişleri için model"""
    from_module: str
    to_module: str
    condition: Optional[str]
    required_data: Optional[List[str]]
    transition_data: Optional[Dict[str, Any]]

class ModuleCapability(TypedDict):
    """Modül yetenekleri tanımı"""
    module_type: ModuleType
    sub_modules: List[str]
    required_permissions: List[str]
    input_requirements: List[str]
    output_types: List[str]
    can_chain_to: List[ModuleType]

class ERPWorkflowConfig(TypedDict):
    """ERP iş akışı konfigürasyonu"""
    available_modules: List[ModuleCapability]
    default_transitions: List[WorkflowTransition]
    user_permissions: Dict[str, List[str]]
    module_timeouts: Dict[str, int]
    
# ==============================================================================
# PARAMETER VALIDATION MODELS
# ==============================================================================

class ParameterRule(TypedDict):
    """Parametre validasyon kuralları"""
    parameter_name: str
    required: bool
    data_type: str
    valid_values: Optional[List[str]]
    validation_regex: Optional[str]
    depends_on: Optional[List[str]]
    error_message: str

class ModuleParameterConfig(TypedDict):
    """Modül parametre konfigürasyonu"""
    module_type: ModuleType
    sub_module: Optional[str]
    parameters: List[ParameterRule]
    parameter_collection_order: List[str]
    conditional_parameters: Optional[Dict[str, List[str]]]

# ==============================================================================
# RESPONSE FORMATTING MODELS
# ==============================================================================

class ButtonAction(TypedDict):
    """Buton aksiyonu tanımı"""
    label: str
    action_type: str  # "module_select", "parameter_input", "confirm", "cancel"
    action_data: Optional[Dict[str, Any]]

class FormField(TypedDict):
    """Form alanı tanımı"""
    field_name: str
    field_type: str  # "text", "date", "select", "multiselect", "number"
    label: str
    required: bool
    options: Optional[List[str]]
    default_value: Optional[str]
    validation_rules: Optional[Dict[str, Any]]

class ChatbotResponseData(TypedDict):
    """Chatbot yanıt verisi"""
    message_type: str  # "text", "buttons", "form", "chart", "table", "file"
    content: str
    buttons: Optional[List[ButtonAction]]
    form_fields: Optional[List[FormField]]
    chart_data: Optional[Dict[str, Any]]
    table_data: Optional[Dict[str, Any]]
    file_attachments: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]

# ==============================================================================
# INTEGRATION MODELS
# ==============================================================================

class DatabaseConnection(TypedDict):
    """Veritabanı bağlantı bilgileri"""
    connection_string: str
    database_type: str  # "mssql", "mysql", "postgresql"
    schema_name: Optional[str]
    table_prefix: Optional[str]
    connection_timeout: int

class APIConfiguration(TypedDict):
    """API konfigürasyonu"""
    api_name: str
    base_url: str
    api_key: Optional[str]
    headers: Optional[Dict[str, str]]
    timeout: int
    rate_limit: Optional[int]

class ERPSystemConfig(TypedDict):
    """ERP sistem konfigürasyonu"""
    database_configs: Dict[str, DatabaseConnection]
    api_configs: Dict[str, APIConfiguration]
    module_permissions: Dict[str, List[str]]
    company_settings: CompanySettings
    default_parameters: Dict[str, Any]

# ==============================================================================
# EXAMPLE FACTORY FUNCTIONS
# ==============================================================================

def create_empty_erp_state() -> ERPChatbotState:
    """Boş ERP chatbot state'i oluştur"""
    return ERPChatbotState(
        messages=[],
        user_info=None,
        session_context=None,
        current_module=None,
        current_sub_module=None,
        workflow_step="initial",
        report_parameters=None,
        report_result=None,
        sql_query_info=None,
        visualization_config=None,
        visualization_result=None,
        support_ticket=None,
        support_resolution=None,
        request_info=None,
        request_status=None,
        company_settings=None,
        external_data=None,
        errors=None,
        requires_user_input=False,
        pending_parameters=None,
        conversation_metadata=None,
        performance_metrics=None
    )

def create_sample_user_info() -> UserInfo:
    """Örnek kullanıcı bilgisi oluştur"""
    return UserInfo(
        user_id="user123",
        username="ahmet.yilmaz",
        company_id="forza_company",
        branch_id="ankara_elvankent",
        permissions=["reporting_view", "master_reports", "branch_data"],
        session_id="session_456"
    )

def create_sample_report_parameters() -> ReportParameters:
    """Örnek rapor parametreleri oluştur"""
    return ReportParameters(
        report_type="ciro",
        branch_filter=["ankara_elvankent"],
        date_range=DateRangeInfo(
            start_date="2025-07-01",
            end_date="2025-07-31",
            period_type="monthly"
        ),
        product_filter=None,
        category_filter=None,
        customer_filter=None,
        additional_filters=None
    )

def create_sample_visualization_config() -> VisualizationConfig:
    """Örnek görselleştirme konfigürasyonu oluştur"""
    return VisualizationConfig(
        chart_type=VisualizationType.PIE_CHART,
        title="Şube Bazında Aylık Ciro Dağılımı",
        x_axis="şube_adı",
        y_axis="toplam_ciro",
        color_scheme="viridis",
        show_legend=True,
        chart_options={"responsive": True, "animation": True}
    )