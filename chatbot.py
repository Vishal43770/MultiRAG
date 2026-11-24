import os
import sys
import json
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

# Load environment configuration from local .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
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


def generate_rag_answer(question: str, context: str) -> tuple[str, dict]:
    """Answer technical questions using retrieved document context."""
    system_prompt = (
        "You are an expert technical assistant. "
        "Answer the user's question accurately using ONLY the provided document context below. "
        "Be clear, concise, and structured.\n\n"
        f"Document Context:\n{context}"
    )

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return _parse_response(response)


def generate_general_answer(question: str) -> tuple[str, dict]:
    """Answer casual chat, greetings, or general knowledge without any document chunks."""
    system_prompt = (
        "You are a friendly, helpful AI assistant. "
        "Answer the user's greeting or general question naturally and concisely. "
        "Mention that you can also answer questions about their uploaded technical documents (PostgreSQL, NIST, etc.)."
    )

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
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


def run_chatbot():
    """Terminal-based interactive RAG chat bot."""
    print("=" * 65)
    print("        🐾 TERMINAL RAG CHAT BOT (Groq + pgvector)        ")
    print("=" * 65)

    init_database(1024)
    chunks_count = get_total_chunks()

    local_embed_file = os.path.join(BASE_DIR, "embeddings.json")
    if chunks_count == 0:
        if os.path.exists(local_embed_file):
            print("📦 Database is empty. Auto-importing 'embeddings.json'...")
            import_embeddings_to_db("embeddings.json")
            chunks_count = get_total_chunks()
        else:
            print("⚠️ Warning: No document chunks in database and 'embeddings.json' not found.")

    print(f"📊 Connected to PostgreSQL: {chunks_count:,} chunks available")
    print(f"🤖 LLM Model: {MODEL_NAME}")
    print(f"🎯 Similarity Threshold: {SIMILARITY_THRESHOLD * 100:.0f}%")
    print("💡 Commands: 'exit' or 'quit' to leave, '/export' to download data to JSON\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye! 👋")
            break

        if user_input.lower() == "/export":
            download_data_from_db("embeddings.json")
            continue

        # 1. Generate query embedding vector
        query_vector = get_embedding(user_input, is_query=True)

        # 2. Print question embedding to terminal and save to JSON
        save_and_display_query_embedding(user_input, query_vector, file_path="query_embedding.json")

        # 3. Check database for relevant document context
        print("🔍 Checking database for relevant document context...")
        matches = search_chunks(user_input, top_k=3, min_score=SIMILARITY_THRESHOLD)

        if matches:
            # Document RAG Mode (Found in PDF)
            print("📑 Mode: Document RAG (Relevant PDF context found)")
            for idx, m in enumerate(matches, 1):
                pct = m["score"] * 100
                print(f"   [{idx}] {m['source']} (Match: {pct:.1f}%)")

            context = "\n\n---\n\n".join([f"Source: {m['source']}\n{m['text']}" for m in matches])
            print(f"\n🧠 Generating answer from document context...\n")
            try:
                answer, token_info = generate_rag_answer(user_input, context)
                print("Bot > " + answer)
                print_token_breakdown(token_info, mode="RAG Mode (PDF Context)")
            except Exception as e:
                print(f"❌ Error querying Groq: {e}")

        else:
            # General AI Mode (Not in PDF -> Direct to Groq, Zero Chunks, Minimal Tokens)
            print("💡 Mode: General Assistant (No PDF chunks needed - saving tokens)")
            print(f"🧠 Generating answer using general knowledge...\n")
            try:
                answer, token_info = generate_general_answer(user_input)
                print("Bot > " + answer)
                print_token_breakdown(token_info, mode="General AI Mode")
            except Exception as e:
                print(f"❌ Error querying Groq: {e}")

        print("\n" + "-" * 65 + "\n")


if __name__ == "__main__":
    run_chatbot()
