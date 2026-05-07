import json
import os
from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict = {}
    category: str = "general"

    @abstractmethod
    def run(self, inputs: dict) -> dict:
        pass


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def dispatch(self, tool_name, inputs):
        if tool_name not in self._tools:
            return {"result": None, "error": "unknown tool"}
        return self._tools[tool_name].run(inputs)

    def list_tools(self):
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]


class DrugLookupTool(Tool):
    name = "drug_lookup"
    description = "Look up drug information from a static reference dictionary"
    category = "lookup"
    input_schema = {"drug_name": "str"}

    _db = {
        "aspirin": "Aspirin (acetylsalicylic acid): NSAID used for pain, fever, and anti-platelet therapy. Typical adult dose 325–650 mg every 4–6 hours; low-dose 81 mg daily for cardiovascular prophylaxis.",
        "metformin": "Metformin: First-line oral biguanide for type 2 diabetes. Typical dose 500–2000 mg/day in divided doses with meals.",
        "lisinopril": "Lisinopril: ACE inhibitor used for hypertension, heart failure, and diabetic nephropathy. Typical dose 5–40 mg once daily.",
        "ibuprofen": "Ibuprofen: NSAID used for pain, fever, and inflammation. Typical adult dose 200–800 mg every 4–6 hours; max 3200 mg/day under medical supervision.",
        "amoxicillin": "Amoxicillin: Broad-spectrum penicillin antibiotic. Typical adult dose 250–500 mg every 8 hours or 500–875 mg every 12 hours depending on indication.",
    }

    def run(self, inputs):
        drug = inputs.get("drug_name", "").strip().lower()
        if drug not in self._db:
            return {"result": None, "error": "Drug not found in reference database"}
        return {"result": self._db[drug], "error": None}


class DosageCalculatorTool(Tool):
    name = "dosage_calculator"
    description = "Calculate a weight-based dosage estimate for informational purposes only"
    category = "calculate"
    input_schema = {"drug_name": "str", "weight_kg": "float", "dose_per_kg": "float"}

    def run(self, inputs):
        try:
            drug = inputs["drug_name"]
            weight = float(inputs["weight_kg"])
            dose_per_kg = float(inputs["dose_per_kg"])
            total = round(weight * dose_per_kg, 2)
            return {
                "result": f"Estimated dose for {drug}: {total} mg — for informational use only",
                "error": None,
            }
        except (KeyError, ValueError, TypeError) as e:
            return {"result": None, "error": f"Invalid inputs: {e}"}


class SymptomSummarizerTool(Tool):
    name = "symptom_summarizer"
    description = "Summarize a list of symptoms and flag any that may require urgent attention"
    category = "summarize"
    input_schema = {"symptoms": "str"}

    _urgent_flags = [
        "chest pain",
        "shortness of breath",
        "loss of consciousness",
        "severe bleeding",
        "stroke",
        "seizure",
    ]

    def run(self, inputs):
        raw = inputs.get("symptoms", "")
        symptoms = [s.strip() for s in raw.split(",") if s.strip()]
        flagged = [s for s in symptoms if any(f in s.lower() for f in self._urgent_flags)]
        lines = []
        for s in symptoms:
            if any(f in s.lower() for f in self._urgent_flags):
                lines.append(f"- {s} [URGENT]")
            else:
                lines.append(f"- {s}")
        summary = "Symptom summary:\n" + "\n".join(lines)
        if flagged:
            summary += "\n\nURGENT: The following symptoms may require immediate medical attention: " + ", ".join(flagged)
        return {"result": summary, "error": None}


class MarketDataLookupTool(Tool):
    name = "market_data_lookup"
    description = "Look up general market data from a static reference dictionary"
    category = "lookup"
    input_schema = {"ticker": "str"}

    _db = {
        "AAPL": {"description": "Apple Inc. — consumer electronics, software, and services company.", "sector": "Technology", "note": "This is static reference data only and does not reflect current market prices."},
        "GOOGL": {"description": "Alphabet Inc. — parent company of Google, operating in search, cloud, and advertising.", "sector": "Communication Services", "note": "This is static reference data only and does not reflect current market prices."},
        "MSFT": {"description": "Microsoft Corporation — software, cloud computing, and enterprise services.", "sector": "Technology", "note": "This is static reference data only and does not reflect current market prices."},
        "TSLA": {"description": "Tesla Inc. — electric vehicles, energy storage, and solar products.", "sector": "Consumer Discretionary", "note": "This is static reference data only and does not reflect current market prices."},
        "SPY": {"description": "SPDR S&P 500 ETF Trust — tracks the S&P 500 index, representing 500 large US companies.", "sector": "Broad Market ETF", "note": "This is static reference data only and does not reflect current market prices."},
        "BTC": {"description": "Bitcoin — decentralized digital currency and store of value. High volatility asset.", "sector": "Cryptocurrency", "note": "This is static reference data only and does not reflect current market prices."},
    }

    def run(self, inputs):
        ticker = inputs.get("ticker", "").strip().upper()
        if ticker not in self._db:
            return {"result": None, "error": "Ticker not found in reference database"}
        entry = self._db[ticker]
        result_str = f"{ticker}: {entry['description']} Sector: {entry['sector']}. {entry['note']}"
        return {"result": result_str, "error": None}


class CompoundInterestCalculatorTool(Tool):
    name = "compound_interest_calculator"
    description = "Calculate compound interest for educational purposes only"
    category = "calculate"
    input_schema = {"principal": "float", "annual_rate": "float", "years": "int", "compounds_per_year": "int"}

    def run(self, inputs):
        try:
            principal = float(inputs["principal"])
            annual_rate = float(inputs["annual_rate"])
            years = int(inputs["years"])
            n = int(inputs["compounds_per_year"])
            r = annual_rate / 100
            amount = round(principal * (1 + r / n) ** (n * years), 2)
            return {
                "result": f"Estimated future value: ${amount} after {years} years at {annual_rate}% annual interest compounded {n} times per year — for educational purposes only",
                "error": None,
            }
        except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return {"result": None, "error": f"Invalid inputs: {e}"}


class PortfolioSummarizerTool(Tool):
    name = "portfolio_summarizer"
    description = "Summarize a list of asset holdings and flag any concentration risks for informational purposes"
    category = "summarize"
    input_schema = {"holdings": "str"}

    def run(self, inputs):
        raw = inputs.get("holdings", "")
        holdings = [h.strip() for h in raw.split(",") if h.strip()]
        total = len(holdings)
        if total == 0:
            return {"result": "No holdings provided.", "error": None}

        bag = {}
        for h in holdings:
            bag[h] = bag.get(h, 0) + 1
        lines = []
        warnings = []
        for asset, count in bag.items():
            if count / total > 0.5:
                lines.append(f"- {asset} ({count}/{total}) [CONCENTRATION RISK]")
                warnings.append(asset)
            else:
                lines.append(f"- {asset} ({count}/{total})")
        unique_assets = list(bag.keys())
        if len(unique_assets) == 1:
            warnings.append(unique_assets[0])
            if unique_assets[0] not in [w for w in warnings if w != unique_assets[0]]:
                lines = [f"- {unique_assets[0]} ({total}/{total}) [CONCENTRATION RISK]"]
        summary = "Portfolio holdings summary:\n" + "\n".join(lines)
        if warnings:
            summary += "\n\nCONCENTRATION RISK detected. Diversification strategy should be discussed with a licensed financial advisor."
        else:
            summary += "\n\nNo concentration risk detected. Consult a licensed financial advisor for personalized guidance."
        return {"result": summary, "error": None}


class FAQLookupTool(Tool):
    name = "faq_lookup"
    description = "Look up answers to common customer questions from a static FAQ dictionary"
    category = "lookup"
    input_schema = {"query": "str"}

    _faq = {
        "return policy": "You may return most items within 30 days of purchase for a full refund, provided they are in original condition with proof of purchase. Certain items such as digital downloads are non-returnable.",
        "shipping time": "Standard shipping takes 5–7 business days. Expedited shipping takes 2–3 business days. Express overnight shipping is available at checkout. Delivery times may vary during peak periods.",
        "order cancellation": "Orders can be cancelled within 1 hour of placement. After that window, the order enters fulfillment and cannot be cancelled. Please contact support immediately if you need to cancel.",
        "password reset": "To reset your password, click 'Forgot Password' on the login page and enter your email address. You will receive a reset link within 5 minutes. Check your spam folder if it does not arrive.",
        "payment methods": "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay. All transactions are encrypted and processed securely.",
        "account deletion": "To delete your account, go to Account Settings > Privacy > Delete Account. Note that this action is permanent and cannot be undone. All your data will be removed within 30 days per our privacy policy.",
        "subscription cancellation": "You can cancel your subscription at any time from Account Settings > Subscription > Cancel. You will retain access until the end of your current billing period. No partial refunds are issued for unused time.",
        "contact": "You can reach our support team at support@example.com or by calling 1-800-555-0100 Monday through Friday, 9am–6pm EST. Live chat is also available on our website.",
    }

    def run(self, inputs):
        query = inputs.get("query", "").lower()
        for key, answer in self._faq.items():
            if key in query:
                return {"result": answer, "error": None}
        query_words = set(query.split())
        for key, answer in self._faq.items():
            if set(key.split()) & query_words:
                return {"result": answer, "error": None}
        return {"result": "I was unable to find a specific answer. A human agent can assist further.", "error": None}


class TicketClassifierTool(Tool):
    name = "ticket_classifier"
    description = "Classify a customer support ticket into a category and priority level"
    category = "classify"
    input_schema = {"ticket_text": "str"}

    _category_keywords = {
        "billing": ["charge", "invoice", "payment", "refund", "bill", "overcharged", "fee"],
        "technical": ["broken", "error", "crash", "bug", "not working", "login", "glitch", "slow"],
        "shipping": ["delivery", "shipped", "tracking", "lost", "package", "arrived", "delayed"],
        "account": ["password", "access", "locked", "username", "profile", "email", "sign in"],
    }
    _priority_keywords = {"urgent": ["immediately", "urgent", "cannot access", "outage", "fraud", "emergency", "asap", "critical"]}

    def run(self, inputs):
        text = inputs.get("ticket_text", "").lower()
        hits = {cat: sum(1 for kw in kws if kw in text) for cat, kws in self._category_keywords.items()}
        winner = max(hits, key=hits.get)
        if hits[winner] == 0:
            winner = "general"
        priority = "urgent" if any(kw in text for kw in self._priority_keywords["urgent"]) else "normal"
        return {"result": f"Category: {winner} | Priority: {priority}", "error": None}


class ConversationSummarizerTool(Tool):
    name = "conversation_summarizer"
    description = "Summarize the conversation history and extract the core customer issue"
    category = "summarize"
    input_schema = {"history": "str"}

    def run(self, inputs):
        history = inputs.get("history", "")
        lines = [l.strip() for l in history.splitlines() if l.strip()]
        all_words = {word for line in lines for word in line.lower().split()}
        questions = [l for l in lines if "?" in l][:3]
        question_block = "\n".join(f"  - {q}" for q in questions) if questions else "  None identified."
        result = (
            f"Conversation turns: {len(lines)}\n"
            f"Unique word count: {len(all_words)}\n"
            f"Identified customer questions:\n{question_block}\n"
            "Full summary should be reviewed by a human agent for complex cases."
        )
        return {"result": result, "error": None}


# ---------------------------------------------------------------------------
# RAG tools (BM25 retrieval over downloaded corpora)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class _RAGTool(Tool):
    """Base RAG tool using BM25Okapi retrieval over a JSON Q&A corpus."""
    category = "rag"
    input_schema = {"query": "str", "top_k": "int (optional, default 3)"}
    _corpus_filename: str = ""

    def __init__(self):
        corpus_path = os.path.join(_DATA_DIR, self._corpus_filename)
        with open(corpus_path) as f:
            self._records = json.load(f)
        # Build BM25 index over "question answer" concatenation
        tokenized = [
            (r["question"] + " " + r["answer"]).lower().split()
            for r in self._records
        ]
        self._bm25 = BM25Okapi(tokenized)

    def run(self, inputs):
        query = inputs.get("query", "").strip()
        if not query:
            return {"result": None, "error": "query must be non-empty"}
        top_k = int(inputs.get("top_k", 3))

        # bm25 does the heavy lifting here
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        idx_list = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        chunks = []
        for idx in idx_list:
            rec = self._records[idx]
            chunks.append(f"Q: {rec['question']}\nA: {rec['answer']}")
        if not chunks:
            return {"result": "No relevant passages found.", "error": None}
        result = "\n\n---\n\n".join(chunks)
        return {"result": result, "error": None}


class MedicalRAGTool(_RAGTool):
    name = "medical_rag_search"
    description = (
        "Retrieve relevant medical Q&A passages from a curated corpus using BM25 keyword search. "
        "Use this to ground responses in domain knowledge before answering."
    )
    _corpus_filename = "medical_qa.json"


class FinancialRAGTool(_RAGTool):
    name = "financial_rag_search"
    description = (
        "Retrieve relevant financial Q&A passages from a curated corpus using BM25 keyword search. "
        "Use this to ground responses in domain knowledge before answering."
    )
    _corpus_filename = "financial_qa.json"


class CustomerServiceRAGTool(_RAGTool):
    name = "customer_service_rag_search"
    description = (
        "Retrieve relevant customer service FAQ passages from a curated corpus using BM25 keyword search. "
        "Use this to ground responses in domain knowledge before answering."
    )
    _corpus_filename = "customer_service_qa.json"
