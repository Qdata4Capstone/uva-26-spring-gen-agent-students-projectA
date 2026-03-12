"""Pydantic models for request/response validation and data structures."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Request / Response ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """Incoming analysis request from the frontend."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")


class SSEEvent(BaseModel):
    """Server-Sent Event payload."""

    event: str  # "thinking" | "report" | "error" | "done"
    data: dict


# ── Financial Data Models ─────────────────────────────────────────────
class YearlyMetrics(BaseModel):
    """Financial metrics for a single fiscal year."""

    year: str
    revenue: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    net_income: Optional[float] = None
    net_income_growth_pct: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    stockholders_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None


class FinancialData(BaseModel):
    """Complete financial data for a stock ticker."""

    ticker: str
    company_name: str = ""
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    metrics: list[YearlyMetrics] = []


class NewsArticle(BaseModel):
    """A single news article."""

    title: str = ""
    url: str = ""
    description: Optional[str] = None
    published: Optional[str] = None
    source: Optional[str] = None


class NewsData(BaseModel):
    """News search result set."""

    query: str
    articles: list[NewsArticle] = []
    source: str = "unknown"


# ── Agent State (shared with LangGraph) ──────────────────────────────
class ThinkingStep(BaseModel):
    """A single thinking/progress step emitted by an agent node."""

    node: str
    message: str
    status: str = "progress"  # started | progress | completed | error

