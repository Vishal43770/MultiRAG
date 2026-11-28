-- Active: 1793687049682@@localhost@5433
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
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Context Window & Token Budget Configurations (50% of Llama 3.3 70B's 128k = 65,536 tokens)
TOTAL_CONTEXT_BUDGET = 65536
MAX_RESPONSE_TOKENS = 4096
HEADROOM_USE = 0.70

# Minimum floors below which borrowing is forbidden
MIN_FLOOR = {
    "system": 800,
    "memory": 300,
    "summary": 300,
    "rag": 3000,
    "short_term": 4000,
}

# Base target allocations
BUDGET = {
    "system": 1000,
    "memory": 3000,
    "summary": 4000,
    "rag": 16000,
    "short_term": 28000,
}

SYSTEM_PROMPT_BUDGET = BUDGET["system"]
LONG_TERM_MEMORY_BUDGET = BUDGET["memory"]
SUMMARY_BUDGET = BUDGET["summary"]
RAG_DOCS_BUDGET = BUDGET["rag"]
SHORT_TERM_MSG_BUDGET = BUDGET["short_term"]


def rebalance_tokens(requested: dict) -> dict:
    """
    Dynamically balances token allocations across sections based on actual demand.
    Priority borrowing rule:
      1. Reclaims unused slack into a shared pool.
      2. Adds 70% of available buffer headroom to the pool.
      3. Distributes pool to sections that want more in priority order: rag -> short_term -> memory -> summary.
      4. If still over budget, borrows from donor sections down to their MIN_FLOOR.
    """
    alloc = dict(BUDGET)

    # 1. Reclaim unused slack into the shared pool
    pool = 0
    for k in alloc:
        req = requested.get(k, 0)
        if req < alloc[k]:
            pool += (alloc[k] - req)
            alloc[k] = req

    # 2. Add usable headroom buffer
    fixed_sum = sum(BUDGET.values()) + MAX_RESPONSE_TOKENS
    headroom = max(0, TOTAL_CONTEXT_BUDGET - fixed_sum)
    pool += int(headroom * HEADROOM_USE)

    # 3. Donate pool to sections that want more (in priority order)
    demand_order = ["rag", "short_term", "memory", "summary"]
    for k in demand_order:
        extra_needed = requested.get(k, 0) - alloc[k]
        if extra_needed > 0:
            granted = min(extra_needed, pool)
            alloc[k] += granted
            pool -= granted

    # 4. If still over-demanded, borrow from low-value donor sections down to their MIN_FLOOR
    donors = ["summary", "memory", "rag", "short_term"]
    for k in demand_order:
        extra_needed = requested.get(k, 0) - alloc[k]
        for d in donors:
            if extra_needed <= 0:
                break
            if d == k:
                continue
            spare = alloc[d] - MIN_FLOOR[d]
            if spare > 0:
                borrowed = min(spare, extra_needed)
                alloc[d] -= borrowed
                alloc[k] += borrowed
                extra_needed -= borrowed

    return alloc


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
        Retrieve recent unsummarized messages that fit within the configured token budget.
        Never drops history from the database; avoids duplicating messages already in the summary.
        """
        all_msgs = get_conversation_messages(self.conversation_id)
        if not all_msgs:
            return []

        summary_data = get_conversation_summary(self.conversation_id)
        last_summarized_id = summary_data["last_message_id"] if summary_data else 0
        unsummarized_msgs = [m for m in all_msgs if m["id"] > last_summarized_id]

        selected = []
        accumulated_tokens = 0

        # Traverse backwards from latest to oldest
        for msg in reversed(unsummarized_msgs):
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
        Token-based summarizer:
        Compress older messages into a rolling summary ONLY when the total unsummarized
        messages exceed SHORT_TERM_MSG_BUDGET. It preserves the most recent turns within
        the budget and summarizes only the older overflow.
        """
        all_msgs = get_conversation_messages(self.conversation_id)
        if not all_msgs:
            return

        existing_summary_data = get_conversation_summary(self.conversation_id)
        last_summarized_id = existing_summary_data["last_message_id"] if existing_summary_data else 0
        existing_summary = existing_summary_data["summary"] if existing_summary_data else ""

        # Only look at messages not yet included in the summary
        unsummarized = [m for m in all_msgs if m["id"] > last_summarized_id]
        if not unsummarized:
            return

        # Calculate token counts backwards (from newest to oldest) to protect recent chat
        accumulated_tokens = 0
        split_index = len(unsummarized)  # start assuming all fit in short term

        for idx in range(len(unsummarized) - 1, -1, -1):
            msg = unsummarized[idx]
            t_count = msg.get("token_count") or estimate_tokens(msg["content"])
            if accumulated_tokens + t_count > SHORT_TERM_MSG_BUDGET:
                split_index = idx + 1
                break
            accumulated_tokens += t_count

        # If all unsummarized messages fit inside SHORT_TERM_MSG_BUDGET, no summary needed yet!
        target_batch = unsummarized[:split_index]
        if not target_batch:
            return

        # Only trigger summary call if the batch to summarize has substantial content (> 500 tokens)
        batch_tokens = sum(m.get("token_count") or estimate_tokens(m["content"]) for m in target_batch)
        if batch_tokens < 500:
            return

        chat_transcript = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in target_batch])

        prompt = (
            "You are a conversation summarizer. Update the ongoing conversation summary.\n"
            "Preserve: user goals, technical decisions, key facts, and unresolved topics.\n"
            "Be dense, factual, and concise (under 250 words).\n\n"
        )
        if existing_summary:
            prompt += f"Existing Summary:\n{existing_summary}\n\n"
        prompt += f"New Conversation Segment to assimilate:\n{chat_transcript}\n\nUpdated Summary:"

        try:
            res = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            updated_summary = (res.choices[0].message.content or "").strip()
            new_last_id = target_batch[-1]["id"]
            save_conversation_summary(self.conversation_id, updated_summary, new_last_id)
            print(f"📝 [Rolling Summary Updated]: Absorbed {len(target_batch)} older messages (~{batch_tokens} tokens)")
        except Exception as e:
            print(f"⚠️ Memory summary update skipped: {e}")

    # ==================================================================================
    # 2. Long-Term Semantic Memory (Cross-Conversation)
    # ==================================================================================

    def _extract_long_term_memory(self, user_text: str, assistant_text: str):
        """
        Analyze user interaction to extract durable preferences or technical profile facts.
        Strictly ignores temporary chit-chat or single-turn topics, and deduplicates existing facts.
        """
        if not groq_client or len(user_text.strip()) < 6:
            return

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
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content or "{}")
            if data.get("has_memory") and data.get("fact"):
                fact = data["fact"].strip()
                category = data.get("category", "preference")
                # Generate 1024-dim embedding via local BGE model
                vector = get_embedding(fact, is_query=False)
                # Deduplicate: skip if a very similar memory already exists for this user
                existing = search_user_memories(self.user_id, vector, top_k=1, min_score=0.88)
                if existing:
                    return
                add_user_memory(self.user_id, fact, vector, category)
                print(f"💡 [Long-Term Memory Learned]: {fact}")
        except Exception as e:
            print(f"⚠️ Long-term memory extraction skipped: {e}")

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

        # Measure tokens requested by each component
        raw_system_tokens = estimate_tokens(system_base)
        raw_mem_tokens = sum(estimate_tokens(m) for m in long_term_mems)
        raw_summary_tokens = estimate_tokens(summary_text)
        raw_rag_tokens = estimate_tokens(rag_context)

        # Estimate unsummarized chat tokens currently in database
        all_msgs = get_conversation_messages(memory_manager.conversation_id)
        last_summarized_id = summary_data["last_message_id"] if summary_data else 0
        unsummarized = [m for m in all_msgs if m["id"] > last_summarized_id]
        raw_chat_tokens = sum(m.get("token_count") or estimate_tokens(m["content"]) for m in unsummarized)

        # Run dynamic priority rebalancing
        allocations = rebalance_tokens({
            "system": raw_system_tokens,
            "memory": raw_mem_tokens,
            "summary": raw_summary_tokens,
            "rag": raw_rag_tokens,
            "short_term": raw_chat_tokens,
        })

        # Construct System Instructions with Memory
        system_sections = [system_base]

        if long_term_mems:
            mems_block = "\n".join([f"• {m}" for m in long_term_mems])
            max_mem_chars = allocations["memory"] * 4
            if len(mems_block) > max_mem_chars:
                mems_block = mems_block[:max_mem_chars] + "..."
            system_sections.append(f"\n[Long-Term User Profile & Preferences]:\n{mems_block}")

        if summary_text:
            max_summary_chars = allocations["summary"] * 4
            if len(summary_text) > max_summary_chars:
                summary_text = summary_text[:max_summary_chars] + "..."
            system_sections.append(f"\n[Summary of Earlier Conversation]:\n{summary_text}")

        if rag_context:
            max_rag_chars = allocations["rag"] * 4
            if len(rag_context) > max_rag_chars:
                rag_context = rag_context[:max_rag_chars] + "\n...[truncated to fit context budget]"
            system_sections.append(f"\n[Retrieved Document Knowledge Context]:\n{rag_context}")

        full_system_prompt = "\n".join(system_sections)

        messages = [{"role": "system", "content": full_system_prompt}]

        # 3. Add active short-term messages using dynamically expanded/borrowed budget
        recent_messages = memory_manager.get_active_context_messages(budget_tokens=allocations["short_term"])
        for msg in recent_messages:
            messages.append(msg)

        # 4. Add current query
        messages.append({"role": "user", "content": query})

        return messages
