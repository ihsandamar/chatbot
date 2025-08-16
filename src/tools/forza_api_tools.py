import requests
import json
import base64
from langchain_core.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from src.models.models import State
from langchain_core.tools.base import BaseToolkit
from langchain_core.tools import BaseTool, tool
from typing import Optional, List


class MasterReportCommonFilterInputModel(BaseModel):
    """Master Report için ortak filtre parametreleri modeli"""
    
    start_date: Optional[str] = Field(None, description="Başlangıç tarihi (YYYY-MM-DD formatında)")
    end_date: Optional[str] = Field(None, description="Bitiş tarihi (YYYY-MM-DD formatında)")
    business_id: str = Field(..., description="İşletme ID'si (zorunlu)")
    start_daily_time_range: Optional[str] = Field(None, description="Günlük başlangıç saati (HH:mm:ss formatında)")
    end_daily_time_range: Optional[str] = Field(None, description="Günlük bitiş saati (HH:mm:ss formatında)")
    branch_ids: Optional[List[str]] = Field(None, description="Şube ID'leri listesi")
    product_ids: Optional[List[str]] = Field(None, description="Ürün ID'leri listesi")
    category_ids: Optional[List[str]] = Field(None, description="Kategori ID'leri listesi")
    dates: Optional[List[str]] = Field(None, description="Özel tarihler listesi")
    day_of_week: Optional[List[int]] = Field(None, description="Haftanın günleri (1-7 arası)")
    platforms: Optional[List[str]] = Field(None, description="Platform listesi")
    payment_type_ids: Optional[List[int]] = Field(None, description="Ödeme tipi ID'leri")
    bank_card_type_ids: Optional[List[str]] = Field(None, description="Banka kartı tipi ID'leri")
    amount_range_first_value: Optional[float] = Field(None, description="Tutar aralığı başlangıç değeri")
    amount_range_second_value: Optional[float] = Field(None, description="Tutar aralığı bitiş değeri")
    quantity_range_first_value: Optional[float] = Field(None, description="Miktar aralığı başlangıç değeri")
    quantity_range_second_value: Optional[float] = Field(None, description="Miktar aralığı bitiş değeri")
    discount_range_first_value: Optional[float] = Field(None, description="İndirim aralığı başlangıç değeri")
    discount_range_second_value: Optional[float] = Field(None, description="İndirim aralığı bitiş değeri")
    currency_unit_ids: Optional[List[str]] = Field(None, description="Para birimi ID'leri")
    payment_user_ids: Optional[List[str]] = Field(None, description="Ödeme kullanıcısı ID'leri")
    documents_kind_ids: Optional[List[str]] = Field(None, description="Doküman türü ID'leri")
    document_type_ids: Optional[List[str]] = Field(None, description="Doküman tipi ID'leri")
    cashier_ids: Optional[List[str]] = Field(None, description="Kasiyer ID'leri")
    customer_ids: Optional[List[str]] = Field(None, description="Müşteri ID'leri")
    methods: Optional[List[str]] = Field(None, description="Metodlar listesi")
    luncheon_voucher_types: Optional[List[str]] = Field(None, description="Yemek çeki türleri")
    table_numbers: Optional[List[str]] = Field(None, description="Masa numaraları")
    search_string: Optional[str] = Field(None, description="Arama metni")
    is_include_opened_table: Optional[bool] = Field(None, description="Açık masaları dahil et")
    is_include_online: Optional[bool] = Field(None, description="Online siparişleri dahil et")
    is_include_food_delivery: Optional[bool] = Field(None, description="Yemek teslimatını dahil et")
    is_income_expense_calculated: Optional[bool] = Field(None, description="Gelir-gider hesaplaması dahil")
    branch_cities: Optional[List[str]] = Field(None, description="Şube şehirleri")
    branch_towns: Optional[List[str]] = Field(None, description="Şube ilçeleri")
    branch_group_names: Optional[List[str]] = Field(None, description="Şube grup isimleri")
    business_types: Optional[List[str]] = Field(None, description="İşletme türleri")
    document_ids: Optional[List[str]] = Field(None, description="Doküman ID'leri")
    is_only_online_delivery: Optional[bool] = Field(None, description="Sadece online teslimat")
    is_include_points_to_total_amount: Optional[bool] = Field(True, description="Puanları toplam tutara dahil et (varsayılan: True)")
    is_only_food_delivery: Optional[bool] = Field(False, description="Sadece yemek teslimatı (varsayılan: False)")


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """JWT token'ı decode eder ve payload'u döner"""
    try:
        # JWT token format: header.payload.signature
        # Base64 decode için padding ekle
        parts = token.split('.')
        if len(parts) != 3:
            return {"error": "Invalid JWT token format"}
        
        payload = parts[1]
        # Base64 padding ekle
        payload += '=' * (4 - len(payload) % 4)
        
        # Base64 decode
        decoded_bytes = base64.b64decode(payload)
        decoded_payload = json.loads(decoded_bytes.decode('utf-8'))
        
        return decoded_payload
        
    except Exception as e:
        return {"error": f"Token decode error: {str(e)}"}


def _create_login_tool(base_url: str):
    @tool
    def login(email: str, password: str) -> Dict[str, Any]:
        """FORZA ERP kullanıcı girişi için endpoint"""
        
        login_url = f"{base_url}/api/auth/login"
        
        payload = {
            "email": email,
            "password": password
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                login_url,
                data=json.dumps(payload),
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Token'ı decode et ve userId'yi ekle
            if "token" in response_data:
                decoded_token = decode_jwt_token(response_data["token"])
                if "error" not in decoded_token:
                    # JWT'den userId'yi al
                    user_id_key = "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier"
                    if user_id_key in decoded_token:
                        response_data["userId"] = decoded_token[user_id_key]
                        response_data["decoded_token"] = decoded_token
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    return login


def _create_get_businesses_tool(base_url: str):
    @tool
    def get_businesses_by_user_id(user_id: str, token: str) -> Dict[str, Any]:
        """Kullanıcı ID'sine göre işletmeleri getir"""
        
        endpoint = f"{base_url}/api/Businesses/getlistbyuserid"
        
        params = {
            "userId": user_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response.json()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    return get_businesses_by_user_id


def _create_get_branches_tool(base_url: str):
    @tool
    def get_branches_by_business_id(business_id: str, token: str) -> Dict[str, Any]:
        """İşletme ID'sine göre şubeleri getir"""
        
        endpoint = f"{base_url}/api/branches/getlistbybusinessid"
        
        params = {
            "businessId": business_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response.json()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    return get_branches_by_business_id


def _create_get_user_branches_tool(base_url: str):
    @tool
    def get_user_branches(email: str, password: str) -> Dict[str, Any]:
        """Kullanıcının tüm şubelerini getir (login -> businesses -> branches chain)"""
        
        # Step 1: Login
        login_tool = _create_login_tool(base_url)
        login_result = login_tool.invoke({"email": email, "password": password})
        
        if login_result["status"] != "success":
            return {
                "status": "error",
                "error": "Login failed",
                "details": login_result
            }
        
        # Extract user_id and token from login response
        try:
            user_id = login_result["data"]["userId"]
            token = login_result["data"]["token"]
        except KeyError as e:
            return {
                "status": "error", 
                "error": f"Missing field in login response: {str(e)}",
                "details": login_result
            }
        
        # Step 2: Get businesses
        businesses_tool = _create_get_businesses_tool(base_url)
        businesses_result = businesses_tool.invoke({"user_id": user_id, "token": token})
        
        if businesses_result["status"] != "success":
            return {
                "status": "error",
                "error": "Failed to get businesses",
                "details": businesses_result
            }
        
        # Step 3: Get branches for all businesses
        branches_tool = _create_get_branches_tool(base_url)
        all_branches = []
        
        try:
            businesses = businesses_result["data"]
            for business in businesses:
                business_id = business["id"]
                branches_result = branches_tool.invoke({"business_id": business_id, "token": token})
                
                if branches_result["status"] == "success":
                    branches_data = {
                        "business_id": business_id,
                        "business_name": business.get("name", "Unknown"),
                        "branches": branches_result["data"]
                    }
                    all_branches.append(branches_data)
                    
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to process branches: {str(e)}",
                "partial_data": all_branches
            }
        
        return {
            "status": "success",
            "data": {
                "user_id": user_id,
                "businesses_count": len(businesses_result["data"]),
                "branches": all_branches
            }
        }
    
    return get_user_branches


def _create_get_total_amount_master_report_tool(base_url: str):
    @tool
    def get_total_amount_master_report(
        business_id: str,
        token: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_daily_time_range: Optional[str] = None,
        end_daily_time_range: Optional[str] = None,
        branch_ids: Optional[List[str]] = None,
        product_ids: Optional[List[str]] = None,
        category_ids: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        day_of_week: Optional[List[int]] = None,
        platforms: Optional[List[str]] = None,
        payment_type_ids: Optional[List[int]] = None,
        bank_card_type_ids: Optional[List[str]] = None,
        amount_range_first_value: Optional[float] = None,
        amount_range_second_value: Optional[float] = None,
        quantity_range_first_value: Optional[float] = None,
        quantity_range_second_value: Optional[float] = None,
        discount_range_first_value: Optional[float] = None,
        discount_range_second_value: Optional[float] = None,
        currency_unit_ids: Optional[List[str]] = None,
        payment_user_ids: Optional[List[str]] = None,
        documents_kind_ids: Optional[List[str]] = None,
        document_type_ids: Optional[List[str]] = None,
        cashier_ids: Optional[List[str]] = None,
        customer_ids: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        luncheon_voucher_types: Optional[List[str]] = None,
        table_numbers: Optional[List[str]] = None,
        search_string: Optional[str] = None,
        is_include_opened_table: Optional[bool] = None,
        is_include_online: Optional[bool] = None,
        is_include_food_delivery: Optional[bool] = None,
        is_income_expense_calculated: Optional[bool] = None,
        branch_cities: Optional[List[str]] = None,
        branch_towns: Optional[List[str]] = None,
        branch_group_names: Optional[List[str]] = None,
        business_types: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        is_only_online_delivery: Optional[bool] = None,
        is_include_points_to_total_amount: Optional[bool] = True,
        is_only_food_delivery: Optional[bool] = False
    ) -> Dict[str, Any]:
        """Şefim ciro raporu"""
        
        endpoint = f"{base_url}/api/MasterReports/gettotalamount"
        
        payload = {
            "startDate": start_date,
            "endDate": end_date,
            "businessId": business_id,
            "startDailyTimeRange": start_daily_time_range,
            "endDailyTimeRange": end_daily_time_range,
            "branchIds": branch_ids,
            "productIds": product_ids,
            "categoryIds": category_ids,
            "dates": dates,
            "dayOfWeek": day_of_week,
            "platforms": platforms,
            "paymentTypeIds": payment_type_ids,
            "bankCardTypeIds": bank_card_type_ids,
            "amountRangeFirstValue": amount_range_first_value,
            "amountRangeSecendValue": amount_range_second_value,
            "quantityRangeFirstValue": quantity_range_first_value,
            "quantityRangeSecendValue": quantity_range_second_value,
            "discountRangeFirstValue": discount_range_first_value,
            "discountRangeSecendValue": discount_range_second_value,
            "currencyUnitIds": currency_unit_ids,
            "paymentUserIds": payment_user_ids,
            "documentsKindIds": documents_kind_ids,
            "documentTypeIds": document_type_ids,
            "cashierIds": cashier_ids,
            "customerIds": customer_ids,
            "methods": methods,
            "luncheonVoucherTypes": luncheon_voucher_types,
            "tableNumbers": table_numbers,
            "searchString": search_string,
            "isIncludeOpenedTable": is_include_opened_table,
            "isIncludeOnline": is_include_online,
            "isIncludeFoodDelivery": is_include_food_delivery,
            "isIncomeExpenseCalculeted": is_income_expense_calculated,
            "branchCities": branch_cities,
            "branchTowns": branch_towns,
            "branchGroupNames": branch_group_names,
            "businessTypes": business_types,
            "documentIds": document_ids,
            "isOnlyOnlineDelivery": is_only_online_delivery,
            "isIncludePointsToTotalAmount": is_include_points_to_total_amount,
            "isOnlyFoodDelivery": is_only_food_delivery
        }
        
        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        try:
            response = requests.post(
                endpoint,
                data=json.dumps(payload),
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Truncate large responses to prevent context length issues
            if isinstance(response_data, dict) and len(str(response_data)) > 5000:
                # Keep only essential fields and truncate data arrays
                summary = {}
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > 10:
                        summary[key] = {
                            "count": len(value),
                            "first_5_items": value[:5],
                            "last_5_items": value[-5:] if len(value) >= 5 else [],
                            "truncated": True
                        }
                    elif isinstance(value, str) and len(value) > 500:
                        summary[key] = value[:500] + "... (truncated)"
                    else:
                        summary[key] = value
                
                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "data": summary,
                    "original_size": len(str(response_data)),
                    "truncated": True
                }
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "data": response_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    return get_total_amount_master_report


class ForzaAPIToolkit(BaseToolkit):
    """Toolkit for Forza ERP API operations. Provides comprehensive ERP functionality including user authentication, business management, and branch operations. Can chain multiple API calls automatically."""
    
    base_url: str = Field(description="Base URL for Forza API")
    
    def __init__(self, base_url: str):
        super().__init__(
            name="ForzaAPIToolkit", 
            description="Comprehensive Forza ERP toolkit for authentication, business and branch management with automatic chaining capabilities.",
            base_url=base_url
        )
    
    def get_tools(self) -> list[BaseTool]:
        """Get the tools available in this toolkit."""
        return [
            _create_login_tool(self.base_url),
            _create_get_businesses_tool(self.base_url),
            _create_get_branches_tool(self.base_url),
            _create_get_total_amount_master_report_tool(self.base_url)
            # _create_get_user_branches_tool(self.base_url)
        ]