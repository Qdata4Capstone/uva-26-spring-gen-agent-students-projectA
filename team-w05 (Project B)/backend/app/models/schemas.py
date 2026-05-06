"""
Pydantic models for request validation and response serialisation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    url: str


class ClinicalTrialSummary(BaseModel):
    """Subset of ClinicalTrials.gov fields exposed to the client."""

    nct_id: str
    title: str
    brief_description: str = ""
    status: str
    recruiting: bool = False
    url: str


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    history: List[ConversationMessage] = []
    user_profile: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    is_crisis: bool = False
    pubmed_articles: List[PubMedArticle] = []
    clinical_trials: List[ClinicalTrialSummary] = []
    pubmed_query_used: Optional[str] = None


# ---------------------------------------------------------------------------
# PubMed search endpoint (exposed separately for transparency)
# ---------------------------------------------------------------------------

class PubMedSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class PubMedSearchResponse(BaseModel):
    query: str
    articles: List[PubMedArticle]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
