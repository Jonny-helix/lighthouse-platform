"""
lighthouse/config.py — Minimal configuration for LIGHTHOUSE.

Provides API key retrieval, default model selection, and contact info.
Equivalent to banyan/config.py but stripped to essentials — no tier system,
no Streamlit dependency.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_EMAIL = "jon@woodspringpartners.com"


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_OPUS_MODEL = "claude-opus-4-6"


def get_api_key() -> str:
    """Return the Anthropic API key from the environment.

    Raises EnvironmentError when the key is missing so callers get a clear
    message rather than a cryptic 401 from the Anthropic SDK.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.  Copy .env.example to .env and "
            "add your key, or export ANTHROPIC_API_KEY in your shell."
        )
    return key


def get_model(purpose: str = "extraction") -> str:
    """Return the Claude model name for a given purpose.

    Args:
        purpose: One of 'extraction', 'query', 'synthesis'.
                 All currently resolve to the same default model.

    Returns:
        Model identifier string suitable for the Anthropic SDK.
    """
    # Allow an env-var override for testing / cost control
    override = os.environ.get("LIGHTHOUSE_MODEL", "").strip()
    if override:
        return override
    return _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Valid categories (coaching domain) — re-exported for convenience
# ---------------------------------------------------------------------------
VALID_CATEGORIES = [
    "Framework",
    "Technique",
    "Principle",
    "Research Finding",
    "Assessment Tool",
    "Case Pattern",
    "Supervision Insight",
    "Contraindication",
]
