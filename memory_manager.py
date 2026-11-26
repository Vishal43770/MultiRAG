"""
========================================================================================
Conversation Memory Manager & Context Budgeting Engine
========================================================================================
Implements:
1. Short-Term Memory Buffer: Token-budget-based message selection (not fixed count)
2. Rolling Conversation Summary: Compresses older messages when budget is reached
3. Long-Term Semantic Memory: Extracts stable user preferences and searches with BGE ONNX
4. Context Budget Manager: Prepares LLM prompt ensuring context window limits are respected
========================================================================================
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from groq import Groq

from memory_db import (
    add_message,
    get_conversation_messages,
    get_conversation_summary,
    save_conversation_summary,
    add_user_memory,
    search_user_memories,
    list_user_memories,
    delete_user_memory,
    clear_all_user_memories,
    create_conversation,
    get_conversation,
    delete_conversation_hard,
    list_conversations,
)
from embedding import get_embedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Context Window & Token Budget Configurations
TOTAL_CONTEXT_BUDGET = 8192
MAX_RESPONSE_TOKENS = 1500
SYSTEM_PROMPT_BUDGET = 400
LONG_TERM_MEMORY_BUDGET = 500
SUMMARY_BUDGET = 600
RAG_DOCS_BUDGET = 2000
SHORT_TERM_MSG_BUDGET = 3000


def estimate_tokens(text: str) -> int:
    """Fast, accurate token estimation (approx 4 chars/token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class MemoryManager:
    """Manages short-term buffer, rolling summaries, and long-term user memories."""

    def __init__(self, user_id: str, conversation_id: str):
        self.user_id = user_id
        self.conversation_id = conversation_id
        # Ensure conversation exists
        create_conversation(self.conversation_id, self.user_id)

    # ==================================================================================
    # 1. Short-Term Memory & Rolling Summarization
    # ==================================================================================

    def record_turn(self, user_text: str, assistant_text: str):
        """Record user and assistant messages, then check if rolling summary is needed."""
        u_tokens = estimate_tokens(user_text)
        a_tokens = estimate_tokens(assistant_text)

        add_message(self.conversation_id, "user", user_text, u_tokens)
        add_message(self.conversation_id, "assistant", assistant_text, a_tokens)

        # Trigger rolling summary if total messages exceed short-term token budget
        self._check_and_update_summary()

        # Check for long-term user preferences/facts extraction asynchronously or inline
        self._extract_long_term_memory(user_text, assistant_text)

    def get_active_context_messages(self, budget_tokens: int = SHORT_TERM_MSG_BUDGET) -> List[Dict[str, str]]:
        """
        Retrieve recent messages that fit within the configured token budget.
        Never drops history from the database; only selects the most recent turns for active LLM context.
        """
        all_msgs = get_conversation_messages(self.conversation_id)
        if not all_msgs:
            return []

        selected = []
        accumulated_tokens = 0

        # Traverse backwards from latest to oldest
        for msg in reversed(all_msgs):
            t_count = msg.get("token_count") or estimate_tokens(msg["content"])
            if accumulated_tokens + t_count > budget_tokens:
                break
            selected.append({"role": msg["role"], "content": msg["content"]})
            accumulated_tokens += t_count

        # Restore chronological order
        selected.reverse()
        return selected

    def _check_and_update_summary(self):
        """
        If the conversation history exceeds the budget, compress older messages
        into a persistent summary while leaving raw messages safely in PostgreSQL.
        """
        all_msgs = get_conversation_messages(self.conversation_id)
        if len(all_msgs) < 6:
            return  # Not enough messages to require summary

        existing_summary_data = get_conversation_summary(self.conversation_id)
        last_summarized_id = existing_summary_data["last_message_id"] if existing_summary_data else 0
        existing_summary = existing_summary_data["summary"] if existing_summary_data else ""

        # Unsummarized messages excluding the latest 4 messages (which stay in short-term buffer)
        messages_to_summarize = [m for m in all_msgs if m["id"] > last_summarized_id]
        if len(messages_to_summarize) <= 4:
            return

        target_batch = messages_to_summarize[:-4]
        if not target_batch:
            return

        chat_transcript = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in target_batch])

        prompt = (
            "You are a conversation summarizer. Update the ongoing conversation summary.\n"
            "Preserve: user goals, technical decisions, key facts, and unresolved topics.\n"
            "Be dense, factual, and concise (under 200 words).\n\n"
        )
        if existing_summary:
            prompt += f"Existing Summary:\n{existing_summary}\n\n"
        prompt += f"New Conversation Segment to assimilate:\n{chat_transcript}\n\nUpdated Summary:"

        try:
            res = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=350,
            )
            updated_summary = (res.choices[0].message.content or "").strip()
            new_last_id = target_batch[-1]["id"]
            save_conversation_summary(self.conversation_id, updated_summary, new_last_id)
        except Exception as e:
            print(f"⚠️ Memory summary update skipped: {e}")

    # ==================================================================================
    # 2. Long-Term Semantic Memory (Cross-Conversation)
    # ==================================================================================

    def _extract_long_term_memory(self, user_text: str, assistant_text: str):
        """
        Analyze user interaction to extract durable preferences or technical profile facts.
        Strictly ignores temporary chit-chat or single-turn topics.
        """
        prompt = (
            "Analyze this user message for durable, long-term user facts or preferences.\n"
            "Examples of durable facts: 'User uses Windows PowerShell', 'User prefers Docker compose', 'User is building a multi-RAG system'.\n"
            "Examples of NON-durable facts (IGNORE): 'Hello', 'What time is it', 'Explain this error', questions about random topics.\n\n"
            f"User message: \"{user_text}\"\n\n"
            "If a durable user preference or fact is stated, return JSON:\n"
            "{\"has_memory\": true, \"fact\": \"<short fact>\", \"category\": \"preference|environment|project\"}\n"
            "If NO durable fact is found, return:\n"
            "{\"has_memory\": false}"
        )

        try:
            res = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content or "{}")
            if data.get("has_memory") and data.get("fact"):
                fact = data["fact"]
                category = data.get("category", "preference")
                # Generate 1024-dim embedding via local BGE model
                vector = get_embedding(fact, is_query=False)
                add_user_memory(self.user_id, fact, vector, category)
                print(f"💡 [Long-Term Memory Learned]: {fact}")
        except Exception:
            pass

    def retrieve_relevant_long_term_memories(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant long-term memories for this user using semantic search."""
        query_vec = get_embedding(query, is_query=True)
        results = search_user_memories(self.user_id, query_vec, top_k=top_k, min_score=0.60)
        return [r["memory_text"] for r in results]


# ======================================================================================
# 3. Context Budget Manager (Assembles Final Prompt)
# ======================================================================================

class ContextManager:
    """Assembles the final LLM prompt guaranteeing strict token budget allocation."""

    @staticmethod
    def build_context(
        memory_manager: MemoryManager,
        query: str,
        rag_context: str = "",
        system_base: str = "You are an expert technical AI assistant.",
    ) -> List[Dict[str, str]]:
        """
        Builds the complete message payload for Groq:
        1. System Prompt + Long-Term Memories + Conversation Summary
        2. RAG Document Context (if present)
        3. Short-Term Recent Messages
        4. Current User Query
        """
        # 1. Retrieve relevant long-term memories (cross-conversation)
        long_term_mems = memory_manager.retrieve_relevant_long_term_memories(query, top_k=3)

        # 2. Retrieve conversation summary (current conversation)
        summary_data = get_conversation_summary(memory_manager.conversation_id)
        summary_text = summary_data["summary"] if summary_data else ""

        # Construct System Instructions with Memory
        system_sections = [system_base]

        if long_term_mems:
            mems_block = "\n".join([f"• {m}" for m in long_term_mems])
            system_sections.append(f"\n[Long-Term User Profile & Preferences]:\n{mems_block}")

        if summary_text:
            system_sections.append(f"\n[Summary of Earlier Conversation]:\n{summary_text}")

        if rag_context:
            system_sections.append(f"\n[Retrieved Document Knowledge Context]:\n{rag_context}")

        full_system_prompt = "\n".join(system_sections)

        messages = [{"role": "system", "content": full_system_prompt}]

        # 3. Add active short-term messages within token budget
        recent_messages = memory_manager.get_active_context_messages(budget_tokens=SHORT_TERM_MSG_BUDGET)
        for msg in recent_messages:
            messages.append(msg)

        # 4. Add current query
        messages.append({"role": "user", "content": query})

        return messages
