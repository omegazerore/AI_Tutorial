from enum import Enum


class Constants(str, Enum):
    """Enumeration of standard constants used in the reporting pipeline.

    Attributes:
        FUTURE_VISION: Project scope for future vision reporting.
        STRATEGY_INPUT: Project scope for strategy input reporting.
        REPORT: Generic report type identifier.
        CSV: File extension for CSV reports.
        JSON: File extension for JSON reports.
        WORD: File extension for Word (.docx) reports.
    """
    FUTURE_VISION = "future_vision"
    STRATEGY_INPUT = "strategy_input"
    REPORT = "report"
    CSV = "csv"
    JSON = "json"
    WORD = "docx"
    TEST_LIMIT = 20