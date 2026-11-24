import os
import json
import psycopg2
import numpy as np
from dotenv import load_dotenv
import onnxruntime as ort
from tokenizers import Tokenizer

# Load environment configuration from local .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")
MODEL_DIR = os.path.join(BASE_DIR, "model")

_tokenizer = None
_session = None


def get_model():
    """Load the ONNX embedding model and tokenizer from the local model folder."""
    global _tokenizer, _session
    if _session is not None:
        return _tokenizer, _session

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Local model directory not found at: {MODEL_DIR}")

    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    onnx_path = os.path.join(MODEL_DIR, "onnx", "model_quantized.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(MODEL_DIR, "onnx", "model.onnx")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

    _tokenizer = Tokenizer.from_file(tokenizer_path)
    _session = ort.InferenceSession(onnx_path)
    return _tokenizer, _session


def get_embedding(text: str, is_query: bool = True) -> list[float]:
    """
    Generate normalized 1024-dimension embedding for input text.
    Queries use the BGE instruction prefix for optimal retrieval accuracy.
    """
    tokenizer, session = get_model()
    input_text = f"Represent this sentence for searching relevant passages: {text}" if is_query else text

    encoding = tokenizer.encode(input_text)
    #print("encoding=",encoding)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    outputs = session.run(None, inputs)
    #print("output=",outputs)
    last_hidden_state = outputs[0]
    #print("last_hidden_state=",last_hidden_state)

    # Mean pooling with attention mask
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    embedding = sum_embeddings / sum_mask

    # L2 normalize
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = (embedding / norm)[0]
    return normalized.tolist()


def get_db_connection():
    """Establish and return connection to Docker PostgreSQL pgvector."""
    return psycopg2.connect(DATABASE_URL)


def init_database(dimensions: int = 1024):
    """Ensure pgvector extension and document_chunks table exist with HNSW index."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({dimensions}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()
    cur.close()
    conn.close()


def download_data_from_db(output_file: str = "embeddings.json") -> int:
    """
    Download/export chunks and embeddings from PostgreSQL into a JSON file in the local folder.
    """
    output_path = os.path.join(BASE_DIR, output_file) if not os.path.isabs(output_file) else output_file
    conn = get_db_connection()
    cur = conn.cursor()
    print(f"📥 Exporting data from Docker PostgreSQL to '{output_path}'...")
    cur.execute("SELECT id, source, text, embedding::text FROM document_chunks ORDER BY id;")
    rows = cur.fetchall()

    documents = []
    for row in rows:
        vec_str = row[3].strip("[]") if row[3] else ""
        vector = [float(v) for v in vec_str.split(",")] if vec_str else []
        documents.append({
            "id": row[0],
            "source": row[1],
            "text": row[2],
            "vector": vector,
        })

    data = {
        "model": "bge-large-en-v1.5",
        "count": len(documents),
        "documents": documents,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    cur.close()
    conn.close()
    print(f"✅ Exported {len(documents)} document chunks to '{output_file}'.")
    return len(documents)


def import_embeddings_to_db(input_file: str = "embeddings.json", batch_size: int = 200) -> int:
    """
    Import chunks and vectors from local JSON file into PostgreSQL pgvector table.
    """
    init_database(1024)
    file_path = os.path.join(BASE_DIR, input_file) if not os.path.isabs(input_file) else input_file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found at: {file_path}")

    print(f"📂 Loading embeddings from '{file_path}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents", [])
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    existing_count = cur.fetchone()[0]
    if existing_count >= len(docs):
        print(f"✅ Database already has {existing_count} chunks. Skipping import.")
        cur.close()
        conn.close()
        return existing_count

    print(f"🚀 Inserting {len(docs)} chunks into PostgreSQL pgvector...")
    insert_sql = """
        INSERT INTO document_chunks (id, source, text, embedding)
        VALUES (%s, %s, %s, %s::vector)
        ON CONFLICT (id) DO NOTHING;
    """

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        values = []
        for doc in batch:
            vec = doc.get("vector") or doc.get("embedding", [])
            values.append((doc["id"], doc.get("source", "unknown"), doc["text"], vec))
        cur.executemany(insert_sql, values)
        conn.commit()
        print(f"   Stored {min(i + batch_size, len(docs))}/{len(docs)} chunks...")

    cur.close()
    conn.close()
    print("✅ All document chunks successfully indexed in PostgreSQL!")
    return len(docs)


def search_chunks(query: str, top_k: int = 3, min_score: float = 0.70) -> list[dict]:
    """
    Search pgvector database using cosine similarity for the user query.
    Default threshold is 0.70 (70%) to prevent false positives for casual chat.
    """
    init_database(1024)
    query_vector = get_embedding(query, is_query=True)

    conn = get_db_connection()
    cur = conn.cursor()

    sql = """
        SELECT id, source, text, 1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cur.execute(sql, (query_vector, query_vector, min_score, query_vector, top_k))
    rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "source": r[1],
            "text": r[2],
            "score": float(r[3]),
        })

    cur.close()
    conn.close()
    return results


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv or "--export" in sys.argv:
        download_data_from_db("embeddings.json")
    elif "--import" in sys.argv:
        import_embeddings_to_db("embeddings.json")
    else:
        print("Testing isolated embedding model & pgvector retrieval (Threshold 0.70):")
        init_database(1024)
        sample_q = "What is NIST SP 800-53?"
        print(f"Query: {sample_q}")
        matches = search_chunks(sample_q, top_k=2, min_score=0.70)
        for idx, m in enumerate(matches, 1):
            print(f"[{idx}] {m['source']} (Score: {m['score']:.4f})")
            print(f"    {m['text'][:120]}...\n")
