# src/tools/custom_date_tool.py
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from datetime import datetime
from src.models import State
from langchain_core.tools.base import BaseToolkit

class DateDiffInput(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

class DateDiffTool(BaseTool):
    name = "custom_date_diff_tool"
    description = "Calculates the number of days between two dates"
    args_schema: Type[BaseModel] = DateDiffInput

    def _run(self, start_date: str, end_date: str) -> str:
        try:
            fmt = "%Y-%m-%d"
            d1 = datetime.strptime(start_date, fmt)
            d2 = datetime.strptime(end_date, fmt)
            delta = (d2 - d1).days
            return f"There are {abs(delta)} days between {start_date} and {end_date}."
        except Exception as e:
            return f"Error parsing dates: {str(e)}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")


class DateToolkit(BaseToolkit):
    def __init__(self):
        self.tools = [
            DateDiffTool()
        ]

    def get_tools(self):
        return self.tools