# LIGHTHOUSE

**Practice Intelligence Platform for Coaching & Personal Development Professionals**

LIGHTHOUSE helps practitioners organise, search, and apply their professional knowledge — frameworks, techniques, research findings, and supervision insights — in a single encrypted knowledge base.

## Phase 1: Practitioner Knowledge Base

- Upload coaching & psychology documents (PDF, DOCX, PPTX)
- AI-powered extraction of frameworks, techniques, principles
- Natural language Q&A over your knowledge base
- Coaching-native taxonomy (8 categories)
- Domain relevance gate for coaching content
- AES-256-GCM encryption with Argon2id key derivation
- `.lighthouse` portable encrypted file format

## Quick Start

```bash
pip install -r requirements.txt
streamlit run lighthouse_app.py
```

## Environment

Copy `.env.example` to `.env` and set your API key:

```
ANTHROPIC_API_KEY=your-key-here
```

## Deployment

Deployed to Render.com. See `render.yaml` for configuration.

---

*Woodspring Partners · March 2026 · Confidential*
