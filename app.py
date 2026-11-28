"""
========================================================================================
🐾 Multi-RAG Chatbot Modern Web UI (Streamlit)
========================================================================================
Features:
1. ChatGPT / Claude Dark Theme aesthetic with interactive chat thread.
2. 3-Layer Memory Integration:
   - Short-term conversation history buffer
   - Rolling conversation summaries
   - Long-term user memories / preferences
3. Hard delete conversation management (with cascade to PostgreSQL).
4. Interactive Similarity Threshold slider (50% - 90%).
5. Rich RAG Inspection:
   - Collapsible cards showing retrieved document passages and similarity scores.
   - Inline visual rendering for retrieved images and diagrams.
   - Token burn & estimated query cost breakdown.
========================================================================================
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Import project modules
from embedding import (
    search_chunks,
    init_database,
    get_embedding,
    get_db_connection,
)
from image_handler import process_image_file
try:
    from pdf_graph_rag import MultimodalPDFExtractor, UnifiedMultimodalChunker, FlatKnowledgeGraph
    PDF_RAG_AVAILABLE = True
except ImportError:
    PDF_RAG_AVAILABLE = False
from memory_db import (
    init_memory_database,
    list_conversations,
    create_conversation,
    delete_conversation_hard,
    get_conversation_messages,
    list_user_memories,
    delete_user_memory,
    clear_all_user_memories,
    get_conversation_summary,
)
from memory_manager import MemoryManager, ContextManager
from guardrails import validate_query_safety, sanitize_input
from model_router import route_query_model, FAST_MODEL, HEAVY_MODEL

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Streamlit Page Setup
st.set_page_config(
    page_title="Multi-RAG Memory Chatbot",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark-mode modern polish + bottom-docked ChatGPT bar)
st.markdown("""
<style>
    /* Global font and subtle styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 8px 14px;
        margin-bottom: 10px;
    }
    .token-badge {
        font-size: 0.78rem;
        color: #8b949e;
        background: #161b22;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #30363d;
        display: inline-block;
        margin-top: 6px;
    }
    .rag-chip {
        font-size: 0.78rem;
        background: #1f2937;
        color: #60a5fa;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #374151;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


def handle_file_ingestion(files):
    """Processes uploaded files and saves embeddings + metadata into pgvector and Neo4j."""
    if not files:
        return
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    with st.status(f"Processing {len(files)} uploaded attachment(s)...", expanded=True) as status:
        for f in files:
            file_path = os.path.join(upload_dir, f.name)
            with open(file_path, "wb") as out:
                out.write(f.getbuffer())

            ext = os.path.splitext(f.name)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                try:
                    process_image_file(file_path)
                    st.write(f"🖼️ Ingested Image: `{f.name}`")
                except Exception as err:
                    st.error(f"Image error ({f.name}): {err}")
            elif ext == ".pdf":
                if PDF_RAG_AVAILABLE:
                    try:
                        extractor = MultimodalPDFExtractor(file_path)
                        chunker = UnifiedMultimodalChunker(chunk_size=1500, overlap=250)
                        chunks = chunker.process_document(extractor)
                        conn = get_db_connection()
                        cur = conn.cursor()
                        insert_sql = """
                            INSERT INTO document_chunks (id, source, text, embedding, metadata)
                            VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                            ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;
                        """
                        import json
                        for ch in chunks:
                            vec = get_embedding(ch["text"], is_query=False)
                            meta = {"modality": "pdf", "source_file": f.name, "page": ch.get("page", 1), "type": ch.get("type", "text")}
                            cur.execute(insert_sql, (ch["id"], f.name, ch["text"], vec, json.dumps(meta)))
                        conn.commit()
                        cur.close()
                        conn.close()

                        graph = FlatKnowledgeGraph()
                        if graph.connect():
                            graph.ingest_chunks(f.name, chunks)
                            graph.close()
                        st.write(f"📑 Ingested PDF: `{f.name}` ({len(chunks)} chunks)")
                    except Exception as err:
                        st.error(f"PDF error ({f.name}): {err}")
            elif ext in [".txt", ".md"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as tf:
                        raw_text = tf.read()
                    conn = get_db_connection()
                    cur = conn.cursor()
                    insert_sql = """
                        INSERT INTO document_chunks (id, source, text, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;
                    """
                    import json
                    step = 1500 - 250
                    c_idx = 1
                    for pos in range(0, len(raw_text), step):
                        piece = raw_text[pos : pos + 1500].strip()
                        if not piece:
                            continue
                        c_id = f"{f.name}_chunk_{c_idx}"
                        vec = get_embedding(piece, is_query=False)
                        meta = {"modality": "text", "source_file": f.name, "chunk_index": c_idx}
                        cur.execute(insert_sql, (c_id, f.name, piece, vec, json.dumps(meta)))
                        c_idx += 1
                    conn.commit()
                    cur.close()
                    conn.close()
                    st.write(f"📝 Ingested Doc: `{f.name}`")
                except Exception as err:
                    st.error(f"Text error ({f.name}): {err}")
        status.update(label="✅ Uploaded files indexed successfully!", state="complete", expanded=False)


# Initialize Databases
@st.cache_resource
def setup_databases():
    init_database(1024)
    init_memory_database(1024)

setup_databases()

# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = "Vishal"

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())[:8]

if "scoped_sources" not in st.session_state:
    st.session_state.scoped_sources = []

# Initialize memory manager
memory_manager = MemoryManager(
    user_id=st.session_state.user_id,
    conversation_id=st.session_state.conversation_id,
)

# ======================================================================================
# SIDEBAR
# ======================================================================================
with st.sidebar:
    st.title("🐾 Multi-RAG Chat")
    st.caption("Hybrid RAG with 3-Layer Conversation Memory")

    # New Chat Button
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        st.session_state.conversation_id = str(uuid.uuid4())[:8]
        st.session_state.scoped_sources = []
        st.rerun()

    st.markdown("---")

    # 1. User Profile Switcher
    current_user = st.text_input("👤 Active User ID", value=st.session_state.user_id)
    if current_user != st.session_state.user_id:
        st.session_state.user_id = current_user
        st.session_state.conversation_id = str(uuid.uuid4())[:8]
        st.session_state.scoped_sources = []
        st.rerun()

    st.markdown("---")

    # 2. Similarity Threshold Slider
    st.markdown("### 🎯 Retrieval Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.35,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="Controls how closely document chunks must match your query. 55% is balanced for short questions & follow-ups. 75%+ is strict.",
    )

    model_choice = st.selectbox(
        "🤖 Model Router",
        ["⚡ Smart Auto-Router", f"🚀 Fast 8B ({FAST_MODEL})", f"🧠 Heavy 70B ({HEAVY_MODEL})"],
        index=0,
        help="Auto-Router dynamically selects 8B for fast lookups/greetings and 70B for deep multi-hop reasoning.",
    )

    st.markdown("---")

    # 3. Conversations Management (Hard Delete Supported)
    st.markdown("### 💬 Your Conversations")
    conversations = list_conversations(st.session_state.user_id)

    if not conversations:
        st.caption("No past conversations.")
    else:
        for c in conversations:
            col1, col2 = st.columns([0.80, 0.20])
            is_active = (c["id"] == st.session_state.conversation_id)
            label = f"{'👉 ' if is_active else ''}Chat #{c['id']} ({c['message_count']} msgs)"
            
            with col1:
                if st.button(label, key=f"select_{c['id']}", use_container_width=True):
                    st.session_state.conversation_id = c["id"]
                    st.session_state.scoped_sources = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{c['id']}", help="Permanently hard-delete conversation"):
                    delete_conversation_hard(c["id"], st.session_state.user_id)
                    if is_active:
                        st.session_state.conversation_id = str(uuid.uuid4())[:8]
                        st.session_state.scoped_sources = []
                    st.rerun()

    st.markdown("---")

    # 4. Long-Term Learned Memories
    st.markdown("### 🧠 Learned Preferences")
    user_mems = list_user_memories(st.session_state.user_id)
    if user_mems:
        for m in user_mems:
            c_mem, c_btn = st.columns([0.85, 0.15])
            with c_mem:
                st.caption(f"• **[{m['category']}]** {m['memory_text']}")
            with c_btn:
                if st.button("x", key=f"del_mem_{m['id']}", help="Delete this memory"):
                    delete_user_memory(m["id"], st.session_state.user_id)
                    st.rerun()
        if st.button("🧹 Clear All Memories", use_container_width=True):
            clear_all_user_memories(st.session_state.user_id)
            st.rerun()
    else:
        st.caption("No long-term memories learned yet. Chat naturally and the AI will remember stable preferences.")


# ======================================================================================
# MAIN CHAT AREA
# ======================================================================================

st.subheader(f"💬 Conversation: `{st.session_state.conversation_id}`")

# Show active file-scope banner if user uploaded external file(s)
if st.session_state.scoped_sources:
    col_scope, col_clear = st.columns([0.78, 0.22])
    with col_scope:
        scoped_list = ", ".join(f"`{s}`" for s in st.session_state.scoped_sources)
        st.info(f"📌 Answering ONLY from uploaded file(s): {scoped_list}")
    with col_clear:
        if st.button("🌐 Search All DB", use_container_width=True):
            st.session_state.scoped_sources = []
            st.rerun()

# Display Conversation Summary Pill if it exists
summary_info = get_conversation_summary(st.session_state.conversation_id)
if summary_info and summary_info.get("summary"):
    with st.expander("📝 Ongoing Conversation Summary (Earlier Turns)"):
        st.info(summary_info["summary"])

# Load all historical messages for this conversation from PostgreSQL
history = get_conversation_messages(st.session_state.conversation_id)

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Native Streamlit Chat Input with built-in '+' / attachment button
user_submission = st.chat_input(
    "Ask Multi-RAG...",
    accept_file="multiple",
    file_type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"],
)

if user_submission:
    user_query = user_submission.text.strip() if user_submission.text else ""
    attached_files = user_submission.files if user_submission.files else []

    # 1. Ingest any attached files first and scope retrieval strictly to them
    if attached_files:
        handle_file_ingestion(attached_files)
        st.session_state.scoped_sources = [f.name for f in attached_files]

    # 2. If user also typed a question, run RAG + LLM
    if not user_query:
        if attached_files:
            file_names = ", ".join(f"`{f.name}`" for f in attached_files)
            st.success(f"✅ Indexed {file_names}. Answers will now focus ONLY on your uploaded file(s)!")
    else:
        # Render User Message immediately
        with st.chat_message("user"):
            if attached_files:
                file_names = ", ".join(f"`{f.name}`" for f in attached_files)
                st.caption(f"📎 Attached: {file_names}")
            st.markdown(user_query)

        # 0. Safety Guardrail Validation
        is_safe, safety_msg = validate_query_safety(user_query)
        if not is_safe:
            with st.chat_message("assistant"):
                st.warning(safety_msg)
            st.stop()

        # Query Vector Search: strictly scoped to uploaded file(s) if active, else full DB
        if st.session_state.scoped_sources:
            matches = search_chunks(
                user_query,
                top_k=5,
                min_score=0.0,
                source_filter=st.session_state.scoped_sources,
            )
        else:
            matches = search_chunks(user_query, top_k=3, min_score=similarity_threshold)
            if not matches and history:
                # Retry with recent user context for short follow-ups (e.g., "give me items list")
                last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
                if last_user_msgs:
                    expanded_query = f"{last_user_msgs[-1]} {user_query}"
                    matches = search_chunks(expanded_query, top_k=3, min_score=similarity_threshold)

        rag_context = ""
        retrieved_images = []
        
        if matches:
            rag_context = "\n\n---\n\n".join([f"Source: {m['source']}\n{m['text']}" for m in matches])
            for m in matches:
                meta = m.get("metadata", {})
                img_path = meta.get("storage_path") or meta.get("image_path")
                if img_path and os.path.exists(img_path) and img_path not in retrieved_images:
                    retrieved_images.append(img_path)

        # 3. Build Prompt via ContextManager
        if rag_context:
            grounding_rule = (
                "Answer using ONLY the provided [Retrieved Document Knowledge Context] and conversation history. "
                "Do NOT invent or guess any numbers, items, or facts not present in the context."
            )
        else:
            grounding_rule = (
                "No matching document chunks were retrieved for this query. "
                "If the user is asking about a specific uploaded document, invoice, or image, clearly say you could not find matching details in the retrieved documents "
                "and suggest lowering the Similarity Threshold or including specific keywords. "
                "NEVER invent, fabricate, or guess numbers, item lists, or document contents."
            )

        sys_instructions = (
            f"You are Multi-RAG, a helpful and expert technical AI assistant with multi-layer conversation memory. "
            f"The user's name is '{st.session_state.user_id}'. Always address them by their name when appropriate. "
            f"{grounding_rule} "
            f"Never call yourself ChatGPT, GPT, or OpenAI. You are Multi-RAG."
        )

        full_messages = ContextManager.build_context(
            memory_manager=memory_manager,
            query=user_query,
            rag_context=rag_context,
            system_base=sys_instructions,
        )

        # 4. Generate LLM Answer
        forced_m = FAST_MODEL if "Fast 8B" in model_choice else (HEAVY_MODEL if "Heavy 70B" in model_choice else None)
        selected_model, route_reason = route_query_model(
            matches=matches,
            has_images=bool(retrieved_images),
            query_length=len(user_query.split()),
            forced_model=forced_m,
        )

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}..."):
                try:
                    response = groq_client.chat.completions.create(
                        model=selected_model,
                        messages=full_messages,
                        temperature=0.2,
                        max_tokens=2048,
                        frequency_penalty=0.1,
                    )
                    answer = response.choices[0].message.content or ""
                    usage = response.usage
                    
                    # Render answer text
                    st.markdown(answer)

                    # Render Retrieved RAG Evidence if available
                    if matches:
                        with st.expander(f"📑 Retrieved Knowledge Context ({len(matches)} chunk(s))"):
                            for idx, m in enumerate(matches, 1):
                                pct = m["score"] * 100
                                st.markdown(f"**[{idx}] {m['source']}** *(Relevance Match: {pct:.1f}%)*")
                                st.text(m["text"][:350] + ("..." if len(m["text"]) > 350 else ""))

                    # Render Images if any were matched
                    if retrieved_images:
                        st.markdown("### 🖼️ Referenced Visual Diagrams")
                        for img_p in retrieved_images:
                            st.image(img_p, caption=os.path.basename(img_p), width="stretch")

                    # Render Token Cost Footnote
                    if usage:
                        in_tokens = usage.prompt_tokens
                        out_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                        cost = ((in_tokens / 1_000_000) * 0.20) + ((out_tokens / 1_000_000) * 0.60)
                        st.markdown(
                            f"<div class='token-badge'>⚡ {total_tokens:,} tokens (In: {in_tokens:,} | Out: {out_tokens:,}) • 🤖 Model: <b>{selected_model}</b> ({route_reason}) • Est. Cost: ~${cost:.5f}</div>",
                            unsafe_allow_html=True,
                        )

                    # 5. Persist turn into short-term buffer, summary, and long-term memory
                    memory_manager.record_turn(user_query, answer)

                except Exception as e:
                    st.error(f"❌ Error communicating with LLM: {e}")
