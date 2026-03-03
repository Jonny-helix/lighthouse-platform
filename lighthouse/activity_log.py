"""
LIGHTHOUSE Activity Log & Cost Tracking.

Chronological event log with API token/cost tracking.
Copied from BANYAN (H30). Domain-agnostic.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# ── Model rates (per token) ─────────────────────────────────────────
# Verified against Anthropic pricing as of February 2026.
MODEL_RATES: Dict[str, Dict[str, float]] = {
    # Current models
    "claude-opus-4-6": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-5-20250929": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000},
    # Legacy models
    "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-opus-4-20250805": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
}

# Fallback rate for unknown models (assume Sonnet-class pricing)
_FALLBACK_RATE = {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000}


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate USD cost for an API call."""
    rates = MODEL_RATES.get(model, _FALLBACK_RATE)
    return (tokens_in * rates["input"]) + (tokens_out * rates["output"])


class ActivityEvent(BaseModel):
    """Single activity event in the chronological log."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = ""
    description: str = ""
    trigger: str = "user"
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    outcome: str = "success"
    detail: Dict[str, Any] = Field(default_factory=dict)


class ActivityLogger:
    """Manages the activity log for a KB session."""

    def __init__(self, events: Optional[List[ActivityEvent]] = None):
        self.events: List[ActivityEvent] = events or []

    def log(
        self,
        event_type: str,
        description: str,
        *,
        trigger: str = "user",
        duration_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str = "",
        outcome: str = "success",
        detail: Optional[Dict[str, Any]] = None,
    ) -> ActivityEvent:
        """Create and append an activity event. Cost is auto-calculated."""
        cost = calculate_cost(model, tokens_in, tokens_out) if model else 0.0
        event = ActivityEvent(
            event_type=event_type,
            description=description,
            trigger=trigger,
            duration_ms=round(duration_ms, 1),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost, 6),
            model=model,
            outcome=outcome,
            detail=detail or {},
        )
        self.events.append(event)
        if len(self.events) > 2000:
            self.events = self.events[-2000:]
        return event

    def filter_by_type(self, event_type: str) -> List[ActivityEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def filter_by_date(self, date_str: str) -> List[ActivityEvent]:
        return [e for e in self.events if e.timestamp.startswith(date_str)]

    def total_cost(self) -> float:
        return round(sum(e.cost_usd for e in self.events), 6)

    def cost_by_type(self) -> Dict[str, float]:
        breakdown: Dict[str, float] = {}
        for e in self.events:
            breakdown[e.event_type] = breakdown.get(e.event_type, 0.0) + e.cost_usd
        return {k: round(v, 6) for k, v in breakdown.items()}

    def cost_by_model(self) -> Dict[str, float]:
        breakdown: Dict[str, float] = {}
        for e in self.events:
            if e.model:
                breakdown[e.model] = breakdown.get(e.model, 0.0) + e.cost_usd
        return {k: round(v, 6) for k, v in breakdown.items()}

    def total_tokens(self) -> Dict[str, int]:
        return {
            "tokens_in": sum(e.tokens_in for e in self.events),
            "tokens_out": sum(e.tokens_out for e in self.events),
        }

    def session_summary(self) -> Dict[str, Any]:
        from datetime import date as date_type
        today = date_type.today().isoformat()
        today_events = self.filter_by_date(today)
        today_cost = sum(e.cost_usd for e in today_events)
        return {
            "total_events": len(self.events),
            "today_events": len(today_events),
            "total_queries": len([e for e in self.events if e.event_type == "query"]),
            "today_queries": len([e for e in today_events if e.event_type == "query"]),
            "total_ingestions": len([e for e in self.events if e.event_type == "ingest"]),
            "today_ingestions": len([e for e in today_events if e.event_type == "ingest"]),
            "total_cost_usd": round(self.total_cost(), 4),
            "today_cost_usd": round(today_cost, 4),
            "cost_by_type": self.cost_by_type(),
            "cost_by_model": self.cost_by_model(),
            "total_tokens": self.total_tokens(),
        }

    def to_list(self) -> List[dict]:
        return [e.model_dump() for e in self.events]

    @classmethod
    def from_list(cls, data: List[dict]) -> "ActivityLogger":
        events = []
        for d in (data or []):
            try:
                events.append(ActivityEvent.model_validate(d))
            except Exception:
                pass
        return cls(events=events)
