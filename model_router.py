"""
========================================================================================
Dynamic LLM Model Router (Clean Score & Data-Driven)
========================================================================================
Routes dynamically based on REAL retrieval signals (scores, chunk counts, images),
with ZERO static regex or hardcoded keyword lists.
========================================================================================
"""

import os
from typing import Tuple, Optional, List, Dict, Any

# Supported Groq Models with Dynamic API Resolution
try:
    from groq import Groq
    _groq_key = os.getenv("GROQ_API_KEY")
    _client = Groq(api_key=_groq_key) if _groq_key else None
    _active_models = [m.id for m in _client.models.list().data] if _client else []
except Exception:
    _active_models = []

def _resolve_model(preferred: str, fallback_keyword: str) -> str:
    if not _active_models:
        return preferred
    if preferred in _active_models:
        return preferred
    for m in _active_models:
        if fallback_keyword in m and "whisper" not in m and "guard" not in m:
            return m
    chat_models = [m for m in _active_models if not any(x in m for x in ["whisper", "guard", "vision", "embed"])]
    return chat_models[0] if chat_models else preferred

FAST_MODEL = _resolve_model(os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"), "8b")
HEAVY_MODEL = _resolve_model(os.getenv("GROQ_HEAVY_MODEL", "llama-3.3-70b-versatile"), "70b")
VISION_MODEL = _resolve_model(os.getenv("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview"), "vision")


def route_query_model(
    matches: List[Dict[str, Any]],
    has_images: bool = False,
    query_length: int = 0,
    forced_model: Optional[str] = None
) -> Tuple[str, str]:
    """
    Data-driven routing based purely on real retrieval signals:
      - Has images? -> Vision Model
      - No chunks retrieved? -> Fast 8B for conversational chat (or 70B if very long complex query)
      - High confidence direct hit (1-2 chunks, score >= 0.75)? -> Fast 8B
      - Multi-chunk synthesis (3+ chunks or lower confidence)? -> Versatile 70B
    """
    if forced_model:
        return forced_model, f"Manually pinned to {forced_model}"

    # 1. Visual input detected
    if has_images:
        return VISION_MODEL, "Visual input detected -> Routed to Vision AI"

    # 2. Zero document chunks retrieved (Pure conversation mode)
    if not matches:
        if query_length > 30:
            return HEAVY_MODEL, "Long complex open-ended query -> Versatile 70B"
        return FAST_MODEL, "General conversation / No document chunks needed -> Fast 8B"

    # 3. Direct factual hit with high confidence
    top_score = matches[0].get("score", 0.0)
    chunk_count = len(matches)

    if chunk_count <= 2 and top_score >= 0.75:
        return FAST_MODEL, f"Direct factual lookup ({top_score*100:.0f}% match) -> Fast 8B (~300ms)"

    # 4. Multi-chunk synthesis or complex multi-document reasoning
    return HEAVY_MODEL, f"Multi-source synthesis ({chunk_count} chunks, {top_score*100:.0f}% top score) -> Versatile 70B"
