# src/tools/custom_date_tool.py
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from src.models import State
from langchain_core.tools.base import BaseToolkit
from langchain_core.tools import BaseTool, tool


@tool
def date_diff(date1: str, date2: str) -> str:
    """Calculate the difference in days between two dates.

    Args:
        date1: First date in YYYY-MM-DD format.
        date2: Second date in YYYY-MM-DD format.
    """
    fmt = "%Y-%m-%d"
    try:
        d1 = datetime.strptime(date1, fmt)
        d2 = datetime.strptime(date2, fmt)
        delta = abs((d2 - d1).days)
        return f"The difference between {date1} and {date2} is {delta} days."
    except ValueError as e:
        return f"Error processing dates: {str(e)}"
    

@tool
def add_days_to_date(date: str, days: int) -> str:
    """Add a number of days to a given date.

    Args:
        date: The original date in YYYY-MM-DD format.
        days: The number of days to add.
    """
    fmt = "%Y-%m-%d"
    try:
        d = datetime.strptime(date, fmt)
        new_date = d + timedelta(days=days)
        return f"The new date after adding {days} days to {date} is {new_date.strftime(fmt)}."
    except ValueError as e:
        return f"Error processing date: {str(e)}"


class DateToolkit(BaseToolkit):
    """Toolkit for date operations."""
    def __init__(self):
        """Initialize the DateToolkit with available tools."""
        super().__init__(name="DateToolkit", description="Toolkit for date operations like calculating date differences and adding days to dates.")

    def get_tools(self) -> list[BaseTool]:
        return [date_diff, add_days_to_date]
    