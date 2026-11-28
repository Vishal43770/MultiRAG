"""
========================================================================================
Safety & Query Validation Guardrails for Multi-RAG
========================================================================================
Features:
1. Prompt Injection & Jailbreak Defense:
   - Blocks attempts to override system instructions ("ignore previous instructions", "DAN mode").
   - Blocks attempts to leak internal system prompts or memory structures.
2. Read-Only Query Validation:
   - Rejects queries attempting destructive SQL or injection commands against the database.
3. Polite Sanitized Responses:
   - Returns a structured status and user-facing explanation when a violation occurs.
========================================================================================
"""

import re
from typing import Tuple, Optional

# Patterns that attempt prompt injection, instruction overrides, or jailbreaks
INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
    r"(?i)\b(system\s+prompt|developer\s+mode|jailbreak|dan\s+mode)\b",
    r"(?i)\b(show|reveal|display|print|output|leak)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions|initial\s+prompt)\b",
    r"(?i)\b(dump|leak|export)\s+(all\s+)?(user\s+memories|database|system_instructions)\b",
    r"(?i)\b(give|show|print)\s+(me\s+)?(the\s+)?long_term_memory_summary\b",
    r"(?i)\byou\s+are\s+now\s+in\s+(unrestricted|god|dan)\s+mode\b",
    r"(?i)\bdo\s+anything\s+now\b",
]

# Patterns attempting destructive or mutating SQL operations
SQL_WRITE_PATTERNS = [
    r"(?i)\b(drop\s+table|delete\s+from|truncate\s+table|insert\s+into|alter\s+table|update\s+\w+\s+set)\b",
    r"(?i)\b(exec\s*\(|execute\s+immediate|shutdown\s+immediate)\b",
    r"(?i)(--|\;)\s*(drop|delete|truncate|insert|update|alter)\b",
]


def validate_query_safety(query: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input query against prompt injection and malicious DB manipulation patterns.
    
    Returns:
        (is_safe, refusal_message)
        If safe: (True, None)
        If unsafe: (False, "Explanation for user")
    """
    if not query or not query.strip():
        return True, None

    cleaned = query.strip()

    # 1. Check for prompt injection or system prompt leaking
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned):
            return (
                False,
                "🛡️ **Security Alert**: Your query triggered our safety guardrails against instruction overrides or internal prompt extraction. Please ask a question related to your documents or task."
            )

    # 2. Check for destructive SQL injection attempts
    for pattern in SQL_WRITE_PATTERNS:
        if re.search(pattern, cleaned):
            return (
                False,
                "🛡️ **Security Alert**: Potentially hazardous database manipulation command detected. Multi-RAG operates in strictly read-only mode."
            )

    return True, None


def sanitize_input(text: str, max_length: int = 4000) -> str:
    """
    Sanitize text input by trimming excessive whitespace and truncating oversized strings.
    """
    if not text:
        return ""
    # Strip null bytes and control chars (except standard newlines/tabs)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    sanitized = re.sub(r"[ \t]+", " ", sanitized)
    return sanitized[:max_length].strip()
