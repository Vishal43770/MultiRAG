import os
import sys
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from embedding import (
    search_chunks,
    get_embedding,
    import_embeddings_to_db,
    download_data_from_db,
    init_database,
    get_db_connection,
)
from memory_db import (
    init_memory_database,
    list_conversations,
    delete_conversation_hard,
    list_user_memories,
    clear_all_user_memories,
)
from memory_manager import MemoryManager, ContextManager
from guardrails import validate_query_safety, sanitize_input
from model_router import route_query_model, FAST_MODEL, HEAVY_MODEL

# Load environment configuration from local .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY is not set in .env")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)

# Session token & cost tracking
SESSION_STATS = {
    "queries": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}

# Pricing per 1 Million tokens (standard open model rate)
INPUT_COST_PER_MILLION = 0.20
OUTPUT_COST_PER_MILLION = 0.60


def get_total_chunks() -> int:
    """Check how many document chunks are currently in PostgreSQL."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except Exception:
        return 0


def save_and_display_query_embedding(query: str, vector: list[float], file_path: str = "query_embedding.json"):
    """
    Print the question's embedding to the terminal and save the full 1024-dim vector to a JSON file.
    """
    preview_first = [round(x, 6) for x in vector[:8]]
    preview_last = [round(x, 6) for x in vector[-4:]]
    norm = sum(x**2 for x in vector) ** 0.5

    print("\n" + "┌" + "─" * 63 + "┐")
    print("│ 🧠 QUESTION EMBEDDING VECTOR (1024 Dimensions)               │")
    print("├" + "─" * 63 + "┤")
    print(f"│  • Dimension Count: {len(vector)} floats (Norm: {norm:.4f})              │")
    print(f"│  • First 8 values:  {preview_first} │")
    print(f"│  • Last 4 values:   {preview_last}                │")
    print("└" + "─" * 63 + "┘")

    payload = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "dimensions": len(vector),
        "normalized_length": round(norm, 6),
        "min_value": round(min(vector), 6),
        "max_value": round(max(vector), 6),
        "embedding": vector,
    }

    full_path = os.path.join(BASE_DIR, file_path) if not os.path.isabs(file_path) else file_path
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"💾 Full embedding vector saved to: '{os.path.basename(full_path)}'\n")


def generate_answer_with_memory(messages: list, model_name: str = None) -> tuple[str, dict]:
    """Generate LLM answer using context-managed messages including short/summary/long-term memory."""
    target_model = model_name or MODEL_NAME
    response = groq_client.chat.completions.create(
        model=target_model,
        messages=messages,
        temperature=0.2,
    )
    return _parse_response(response)


def _parse_response(response) -> tuple[str, dict]:
    """Extract answer and token usage from Groq response."""
    answer = response.choices[0].message.content or ""
    usage = response.usage

    token_info = {
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }

    SESSION_STATS["queries"] += 1
    SESSION_STATS["input_tokens"] += token_info["input_tokens"]
    SESSION_STATS["output_tokens"] += token_info["output_tokens"]
    SESSION_STATS["total_tokens"] += token_info["total_tokens"]

    return answer, token_info


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated dollar cost from burned tokens."""
    in_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    out_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    return in_cost + out_cost


def print_token_breakdown(turn_tokens: dict, mode: str):
    """Display an informative, clean token burn and cost breakdown."""
    turn_in = turn_tokens["input_tokens"]
    turn_out = turn_tokens["output_tokens"]
    turn_total = turn_tokens["total_tokens"]
    turn_cost = calculate_cost(turn_in, turn_out)
    session_cost = calculate_cost(SESSION_STATS["input_tokens"], SESSION_STATS["output_tokens"])

    print("\n" + "┌" + "─" * 58 + "┐")
    print(f"│ 🔥 TOKEN BURN & ESTIMATED COST [{mode:<20} │")
    print("├" + "─" * 58 + "┤")
    print(f"│  • Input (Prompt) Tokens:      {turn_in:>8,} tokens        │")
    print(f"│  • Output (Answer) Tokens:     {turn_out:>8,} tokens        │")
    print(f"│  • Turn Total:                 {turn_total:>8,} tokens (~${turn_cost:.5f}) │")
    print("├" + "─" * 58 + "┤")
    print(f"│  • Session Cumulative ({SESSION_STATS['queries']} msgs): {SESSION_STATS['total_tokens']:>8,} tokens (~${session_cost:.5f}) │")
    print("└" + "─" * 58 + "┘")


def run_chatbot(user_id: str = "default_user", conversation_id: str = None):
    """Terminal-based interactive RAG chat bot with 3-Layer ChatGPT-like Memory."""
    print("=" * 65)
    print("   🐾 TERMINAL RAG CHAT BOT with 3-LAYER MEMORY (PostgreSQL)   ")
    print("=" * 65)

    init_database(1024)
    init_memory_database(1024)
    chunks_count = get_total_chunks()

    local_embed_file = os.path.join(BASE_DIR, "embeddings.json")
    if chunks_count == 0:
        if os.path.exists(local_embed_file):
            print("📦 Database is empty. Auto-importing 'embeddings.json'...")
            import_embeddings_to_db("embeddings.json")
            chunks_count = get_total_chunks()
        else:
            print("⚠️ Warning: No document chunks in database and 'embeddings.json' not found.")

    conv_id = conversation_id or str(uuid.uuid4())[:8]
    memory_manager = MemoryManager(user_id=user_id, conversation_id=conv_id)

    print(f"👤 User ID: {user_id}")
    print(f"💬 Conversation ID: {conv_id}")
    print(f"📊 Knowledge Base: {chunks_count:,} document chunks available")
    print(f"🤖 LLM Model: {MODEL_NAME}")
    print(f"🎯 Similarity Threshold: {SIMILARITY_THRESHOLD * 100:.0f}%")
    print("💡 Commands:")
    print("   /new            -> Start a fresh new conversation")
    print("   /list           -> List your past conversations")
    print("   /load <id>      -> Reopen/switch to an existing conversation")
    print("   /delete <id>    -> HARD DELETE a conversation (purges messages & summary)")
    print("   /memories       -> View your learned long-term preferences")
    print("   /clear-memories -> Purge all your long-term preferences")
    print("   /export         -> Download vector chunks to embeddings.json")
    print("   exit / quit     -> Leave chat\n")

    while True:
        try:
            user_input = input(f"[{conv_id}] You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Goodbye! 👋")
            break

        if cmd == "/new":
            conv_id = str(uuid.uuid4())[:8]
            memory_manager = MemoryManager(user_id=user_id, conversation_id=conv_id)
            print(f"✨ Started fresh conversation: {conv_id}\n")
            continue

        if cmd == "/list":
            convs = list_conversations(user_id)
            print("\n📋 Your Conversations:")
            if not convs:
                print("   (No conversations found)")
            for c in convs:
                marker = "👉 [ACTIVE]" if c["id"] == conv_id else "  "
                print(f"   {marker} ID: {c['id']} | Messages: {c['message_count']} | Last active: {c['updated_at']}")
            print()
            continue

        if cmd.startswith("/load "):
            target_id = user_input.split(maxsplit=1)[1].strip()
            conv_id = target_id
            memory_manager = MemoryManager(user_id=user_id, conversation_id=conv_id)
            print(f"📂 Loaded conversation: {conv_id}\n")
            continue

        if cmd.startswith("/delete "):
            target_id = user_input.split(maxsplit=1)[1].strip()
            success = delete_conversation_hard(target_id, user_id)
            if success:
                print(f"🗑️ Successfully hard-deleted conversation '{target_id}' and all its messages.")
                if target_id == conv_id:
                    conv_id = str(uuid.uuid4())[:8]
                    memory_manager = MemoryManager(user_id=user_id, conversation_id=conv_id)
                    print(f"✨ Switched to new active conversation: {conv_id}")
            else:
                print(f"❌ Could not find or delete conversation '{target_id}'.")
            print()
            continue

        if cmd == "/memories":
            mems = list_user_memories(user_id)
            print(f"\n🧠 Long-Term Memories Stored for User '{user_id}':")
            if not mems:
                print("   (No long-term memories learned yet)")
            for m in mems:
                print(f"   • [{m['category']}] {m['memory_text']} (ID: {m['id']})")
            print()
            continue

        if cmd == "/clear-memories":
            count = clear_all_user_memories(user_id)
            print(f"🧹 Cleared {count} long-term memories for user '{user_id}'.\n")
            continue

        if cmd == "/export":
            download_data_from_db("embeddings.json")
            continue

        # 0. Safety Guardrail Validation
        is_safe, safety_msg = validate_query_safety(user_input)
        if not is_safe:
            print(f"\n{safety_msg}\n")
            continue

        # 1. Generate query embedding vector
        query_vector = get_embedding(user_input, is_query=True)
        save_and_display_query_embedding(user_input, query_vector, file_path="query_embedding.json")

        # 2. Check vector database for relevant document chunks
        print("🔍 Checking database for relevant document context...")
        matches = search_chunks(user_input, top_k=3, min_score=SIMILARITY_THRESHOLD)

        rag_context = ""
        mode_label = "General AI with Memory"
        if matches:
            mode_label = "Document RAG with Memory"
            print("📑 Mode: Document RAG (Relevant context found)")
            for idx, m in enumerate(matches, 1):
                pct = m["score"] * 100
                print(f"   [{idx}] {m['source']} (Match: {pct:.1f}%)")
            rag_context = "\n\n---\n\n".join([f"Source: {m['source']}\n{m['text']}" for m in matches])
        else:
            print("💡 Mode: General Assistant with Memory (No PDF chunks needed)")

        # 3. Assemble prompt with ContextManager (System + Long-Term + Summary + RAG + Short-Term History)
        sys_instructions = (
            "You are an expert technical AI assistant with conversation memory. "
            "Use the provided document context, user profile preferences, and conversation history to answer accurately. "
            "Be structured, concise, and helpful."
        )
        full_messages = ContextManager.build_context(
            memory_manager=memory_manager,
            query=user_input,
            rag_context=rag_context,
            system_base=sys_instructions,
        )

        # 4. Route model & Generate answer from Groq
        selected_model, route_reason = route_query_model(
            matches=matches,
            has_images=False,
            query_length=len(user_input.split()),
        )
        print(f"🤖 Model Router: Selected '{selected_model}' ({route_reason})")
        print(f"\n🧠 Generating answer with active memory context...\n")
        try:
            answer, token_info = generate_answer_with_memory(full_messages, model_name=selected_model)
            print("Bot > " + answer)
            print_token_breakdown(token_info, mode=f"{mode_label} [{selected_model}]")

            # 5. Persist turn into Short-Term Memory, check rolling summary & update long-term memory
            memory_manager.record_turn(user_input, answer)
        except Exception as e:
            print(f"❌ Error querying Groq: {e}")

        print("\n" + "-" * 65 + "\n")


if __name__ == "__main__":
    cli_user = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "default_user"
    run_chatbot(user_id=cli_user)
