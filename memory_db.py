"""
========================================================================================
Conversation Memory Database Layer (PostgreSQL + pgvector)
========================================================================================
Handles schema creation and CRUD operations for:
1. conversations (hard delete with cascade)
2. messages (short-term history log)
3. conversation_summaries (rolling compressed summary)
4. user_memories (cross-conversation long-term semantic memory)
========================================================================================
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")


def get_memory_db_connection():
    """Establish and return connection to PostgreSQL."""
    return psycopg2.connect(DATABASE_URL)


def init_memory_database(dimensions: int = 1024):
    """
    Ensure all tables, constraints, foreign keys with ON DELETE CASCADE,
    and pgvector indexes exist for the memory system.
    """
    conn = get_memory_db_connection()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. Conversations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT DEFAULT 'New Conversation',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);")

    # 2. Messages table (Short-term storage, foreign key with ON DELETE CASCADE)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            token_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);")

    # 3. Conversation Summaries (Rolling compression)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            last_message_id INT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 4. Long-Term Semantic User Memory (Cross-conversation, isolated by user_id)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS user_memories (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_text TEXT NOT NULL,
            category VARCHAR(50) DEFAULT 'general',
            embedding vector({dimensions}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id);")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_memories_embedding_hnsw 
        ON user_memories USING hnsw (embedding vector_cosine_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()


# ======================================================================================
# Conversations CRUD (Hard Delete Supported)
# ======================================================================================

def create_conversation(conversation_id: str, user_id: str, title: str = "New Conversation") -> Dict[str, Any]:
    """Create or ensure a conversation exists for a user without bumping updated_at on read."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        INSERT INTO conversations (id, user_id, title)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
    """, (conversation_id, user_id, title))
    conn.commit()
    cur.execute("SELECT * FROM conversations WHERE id = %s;", (conversation_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row)


def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    """List all active conversations for a given user, ordered by recent update."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
               COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.user_id = %s
        GROUP BY c.id
        ORDER BY c.updated_at DESC;
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a single conversation belonging to user_id."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT * FROM conversations WHERE id = %s AND user_id = %s;
    """, (conversation_id, user_id))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def delete_conversation_hard(conversation_id: str, user_id: str) -> bool:
    """
    Hard delete conversation from PostgreSQL.
    Cascades automatically to delete all linked messages and summaries.
    """
    conn = get_memory_db_connection()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM conversations 
        WHERE id = %s AND user_id = %s;
    """, (conversation_id, user_id))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted


# ======================================================================================
# Messages CRUD (Short-Term Memory)
# ======================================================================================

def add_message(conversation_id: str, role: str, content: str, token_count: int = 0) -> Dict[str, Any]:
    """Store user or assistant message to database and update conversation timestamp."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        INSERT INTO messages (conversation_id, role, content, token_count)
        VALUES (%s, %s, %s, %s)
        RETURNING *;
    """, (conversation_id, role, content, token_count))
    msg = cur.fetchone()

    cur.execute("""
        UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s;
    """, (conversation_id,))

    conn.commit()
    cur.close()
    conn.close()
    return dict(msg)


def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Get all messages in chronological order for a conversation."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, role, content, token_count, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY id ASC;
    """, (conversation_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


# ======================================================================================
# Conversation Summaries CRUD
# ======================================================================================

def get_conversation_summary(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve current summary for conversation."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT summary, last_message_id, updated_at
        FROM conversation_summaries
        WHERE conversation_id = %s;
    """, (conversation_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def save_conversation_summary(conversation_id: str, summary: str, last_message_id: int):
    """Upsert summary for a conversation."""
    conn = get_memory_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversation_summaries (conversation_id, summary, last_message_id, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (conversation_id) DO UPDATE SET
            summary = EXCLUDED.summary,
            last_message_id = EXCLUDED.last_message_id,
            updated_at = CURRENT_TIMESTAMP;
    """, (conversation_id, summary, last_message_id))
    conn.commit()
    cur.close()
    conn.close()


# ======================================================================================
# Long-Term User Memories CRUD (pgvector Semantic Search)
# ======================================================================================

def add_user_memory(user_id: str, memory_text: str, vector: list[float], category: str = "preference") -> int:
    """Store stable user memory / preference with 1024-dim embedding."""
    conn = get_memory_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_memories (user_id, memory_text, category, embedding)
        VALUES (%s, %s, %s, %s::vector)
        RETURNING id;
    """, (user_id, memory_text, category, vector))
    mem_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return mem_id


def search_user_memories(user_id: str, query_vector: list[float], top_k: int = 3, min_score: float = 0.65) -> List[Dict[str, Any]]:
    """Search long-term memories strictly isolated by user_id using cosine similarity."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, memory_text, category, 1 - (embedding <=> %s::vector) AS score
        FROM user_memories
        WHERE user_id = %s
          AND 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_vector, user_id, query_vector, min_score, query_vector, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def list_user_memories(user_id: str) -> List[Dict[str, Any]]:
    """List all long-term memories stored for a user."""
    conn = get_memory_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, memory_text, category, created_at
        FROM user_memories
        WHERE user_id = %s
        ORDER BY id DESC;
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def delete_user_memory(memory_id: int, user_id: str) -> bool:
    """Delete a specific long-term memory for a user."""
    conn = get_memory_db_connection()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM user_memories WHERE id = %s AND user_id = %s;
    """, (memory_id, user_id))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def clear_all_user_memories(user_id: str) -> int:
    """Clear all long-term memories for a user."""
    conn = get_memory_db_connection()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM user_memories WHERE user_id = %s;
    """, (user_id,))
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return count
