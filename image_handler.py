"""
========================================================================================
Multimodal Image Handler & Vision RAG Engine
========================================================================================
Features:
1. Vision AI Analysis (Groq Multimodal Vision with qwen/qwen3.8-27b):
   - Detailed visual technical description
   - OCR / Visible Text transcription (labels, components, numbers, tables)
   - Diagram connection & data flow analysis
2. Standardized Chunking (1500 chars / 250 overlap):
   - Formats visual descriptions into searchable chunks
   - Stores image metadata (dimensions, format, saved path)
3. PostgreSQL pgvector Ingestion:
   - Embeds visual descriptions using 1024-dim BGE-Large ONNX model
   - Inserts directly into 'document_chunks' table
4. Neo4j Knowledge Graph Ingestion:
   - (:Image) -[:HAS_CHUNK]-> (:Chunk) -[:MENTIONS]-> (:Entity)
5. Direct Visual Q&A:
   - Ask any question about an image directly in the terminal
========================================================================================
"""

import os
import sys
import base64
import io
import re
import json
from typing import Dict, Any, List, Optional
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

# Load environment configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

# Import local database and embedding functions
try:
    from embedding import get_embedding, get_db_connection, init_database
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


# ======================================================================================
# 1. Vision AI Analysis & OCR
# ======================================================================================

def encode_image_to_base64(image_path: str, max_size: int = 1536) -> tuple[str, str, int, int]:
    """Load, optionally resize large images, and encode to base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size
        img_format = img.format or "PNG"

        # Downscale if excessively large to save bandwidth while preserving clarity
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            width, height = new_size

        buffer = io.BytesIO()
        img.save(buffer, format=img_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = f"image/{img_format.lower()}"
        return encoded, mime, width, height


def analyze_image_with_vision(image_path: str) -> Dict[str, Any]:
    """
    Use Vision AI to extract comprehensive visual & textual metadata:
    - has_text: boolean indicating whether text is present in the image
    - category: Diagram, Screenshot, Chart, Photo, Illustration, etc.
    - scene_description: Detailed visual scene/component breakdown
    - ocr_text: Verbatim extracted text and labels (or empty string if none)
    - entities: Key technical/real-world entities and tags
    """
    filename = os.path.basename(image_path)
    print(f"👁️ Analyzing image '{filename}' with Vision AI ({VISION_MODEL})...")

    b64_data, mime_type, width, height = encode_image_to_base64(image_path)
    file_size_kb = round(os.path.getsize(image_path) / 1024, 2) if os.path.exists(image_path) else 0

    prompt = (
        "You are an expert multimodal document and image analyzer. Analyze this image thoroughly.\n"
        "Return your answer strictly using the following structured tags:\n\n"
        "[HAS_TEXT]: (Yes or No)\n"
        "[CATEGORY]: (e.g. Architecture Diagram, Flowchart, UI Screenshot, Graph/Chart, Photo/Real-world, Artwork/Illustration)\n"
        "[VISUAL SCENE DESCRIPTION]: (Detailed description of visual contents, layout, objects, colors, flow, and purpose. If no text, describe objects, atmosphere, and scene thoroughly.)\n"
        "[OCR TEXT]: (Verbatim text transcription of every title, label, code, number, or caption visible. If NO text exists, write 'NONE'.)\n"
        "[KEY ENTITIES & TAGS]: (Comma-separated list of 5-12 keywords, concepts, or objects shown)"
    )

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}}
            ]
        }],
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""
    usage = response.usage

    # Extract HAS_TEXT
    has_text_match = re.search(r"\[HAS_TEXT\]:?\s*(.*?)(?=\n\[|\n\n|$)", content, re.IGNORECASE)
    has_text_val = has_text_match.group(1).strip().lower() if has_text_match else ""
    has_text = "yes" in has_text_val or "true" in has_text_val

    # Extract Category
    cat_match = re.search(r"\[CATEGORY\]:?\s*(.*?)(?=\n\[|\n\n|$)", content, re.IGNORECASE)
    category = cat_match.group(1).strip() if cat_match else ("Diagram" if has_text else "Photo / Image")

    # Extract OCR Text
    ocr_match = re.search(r"\[OCR TEXT\]:?\s*(.*?)(?=\n\[|\n\n|$)", content, re.IGNORECASE | re.DOTALL)
    ocr_raw = ocr_match.group(1).strip() if ocr_match else ""
    if ocr_raw.upper() in ["NONE", "N/A", "NO TEXT"]:
        ocr_raw = ""
        has_text = False

    # Extract Visual Description
    desc_match = re.search(r"\[VISUAL SCENE DESCRIPTION\]:?\s*(.*?)(?=\n\[|\n\n|$)", content, re.IGNORECASE | re.DOTALL)
    scene_description = desc_match.group(1).strip() if desc_match else content

    # Extract Key Entities & Tags
    ent_match = re.search(r"\[KEY ENTITIES & TAGS\]:?\s*(.*?)(?=\n\[|\n\n|$)", content, re.IGNORECASE)
    entities = []
    if ent_match:
        entities = [e.strip() for e in ent_match.group(1).split(",") if e.strip()]

    # Construct rich composite text representation for embedding
    composite_text = f"Image: {filename}\nCategory: {category}\nScene Description: {scene_description}\n"
    if has_text and ocr_raw:
        composite_text += f"Visible Text / OCR: {ocr_raw}\n"
    if entities:
        composite_text += f"Tags: {', '.join(entities)}"

    metadata = {
        "modality": "image",
        "has_text": has_text,
        "category": category,
        "source_file": filename,
        "storage_path": os.path.abspath(image_path),
        "dimensions": {
            "width": width,
            "height": height,
            "file_size_kb": file_size_kb,
        },
        "ocr_detected": bool(has_text and ocr_raw),
        "ocr_text_preview": ocr_raw[:300] if ocr_raw else "",
        "tags": entities,
    }

    return {
        "filename": filename,
        "image_path": os.path.abspath(image_path),
        "width": width,
        "height": height,
        "has_text": has_text,
        "category": category,
        "ocr_text": ocr_raw,
        "scene_description": scene_description,
        "full_analysis": composite_text,
        "entities": entities,
        "metadata": metadata,
        "tokens": {
            "input": usage.prompt_tokens if usage else 0,
            "output": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        }
    }


# ======================================================================================
# 2. Chunking (1500 chars / 250 overlap)  16 percentage overlap in chunk size
# ======================================================================================

def chunk_image_analysis(analysis: Dict[str, Any], chunk_size: int = 1500, overlap: int = 250) -> List[Dict[str, Any]]:
    """Convert image description and OCR text into standard 1500/250 chunks with metadata."""
    filename = analysis["filename"]
    category = analysis["category"]
    full_text = analysis["full_analysis"]
    base_metadata = analysis.get("metadata", {})

    chunks = []
    header = f"[Image: {filename} | Category: {category} | HasText: {analysis.get('has_text', False)} | Dimensions: {analysis['width']}x{analysis['height']}px]\n"

    # If text fits inside chunk_size, keep as 1 chunk
    if len(header) + len(full_text) <= chunk_size:
        chunks.append({
            "id": f"{filename}_chunk_1",
            "source": filename,
            "type": "image",
            "image_path": analysis["image_path"],
            "category": category,
            "text": header + full_text,
            "entities": analysis["entities"],
            "metadata": {**base_metadata, "chunk_index": 1, "total_chunks": 1},
        })
        return chunks

    # Otherwise recursively split with overlap
    current_idx = 0
    chunk_counter = 0
    while current_idx < len(full_text):
        chunk_counter += 1
        end_idx = current_idx + (chunk_size - len(header))
        piece = full_text[current_idx:end_idx].strip()

        chunks.append({
            "id": f"{filename}_chunk_{chunk_counter}",
            "source": filename,
            "type": "image",
            "image_path": analysis["image_path"],
            "category": category,
            "text": f"{header}(Part {chunk_counter})\n{piece}",
            "entities": analysis["entities"],
            "metadata": {**base_metadata, "chunk_index": chunk_counter},
        })

        if end_idx >= len(full_text):
            break
        current_idx = end_idx - overlap

    for ch in chunks:
        ch["metadata"]["total_chunks"] = len(chunks)

    return chunks


# ======================================================================================
# 3. Database Ingestion (PostgreSQL pgvector)
# ======================================================================================

def ingest_image_chunks_to_postgres(chunks: List[Dict[str, Any]]) -> int:
    """Store image chunks, 1024-dim embeddings, and metadata into PostgreSQL pgvector."""
    if not EMBEDDING_AVAILABLE:
        print("⚠️ 'embedding.py' not available. Skipping pgvector ingestion.")
        return 0

    init_database(1024)
    conn = get_db_connection()
    cur = conn.cursor()

    insert_sql = """
        INSERT INTO document_chunks (id, source, text, embedding, metadata)
        VALUES (%s, %s, %s, %s::vector, %s::jsonb)
        ON CONFLICT (id) DO UPDATE 
        SET text = EXCLUDED.text,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata;
    """

    print(f"🚀 Indexing {len(chunks)} image chunk(s) into PostgreSQL pgvector...")
    for ch in chunks:
        vector = get_embedding(ch["text"], is_query=False)
        meta_json = json.dumps(ch.get("metadata", {}))
        cur.execute(insert_sql, (ch["id"], ch["source"], ch["text"], vector, meta_json))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Image successfully indexed in PostgreSQL with metadata!")
    return len(chunks)


# ======================================================================================
# 4. Neo4j Graph Ingestion
# ======================================================================================

def ingest_image_to_neo4j(analysis: Dict[str, Any], chunks: List[Dict[str, Any]]):
    """Create (:Image)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity) in Neo4j."""
    if not NEO4J_AVAILABLE:
        return

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "ragpassword")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # 1. Image Node
            session.run("""
                MERGE (img:Image {filename: $filename})
                ON CREATE SET 
                    img.path = $path,
                    img.width = $width,
                    img.height = $height,
                    img.category = $category,
                    img.has_text = $has_text,
                    img.ocr_preview = $ocr_preview;
            """, filename=analysis["filename"], path=analysis["image_path"],
                width=analysis["width"], height=analysis["height"], category=analysis["category"],
                has_text=analysis.get("has_text", False), ocr_preview=analysis.get("ocr_text", "")[:300])

            # 2. Chunk Nodes & Entities
            for ch in chunks:
                session.run("""
                    MATCH (img:Image {filename: $filename})
                    MERGE (c:Chunk {id: $c_id})
                    ON CREATE SET 
                        c.type = 'image',
                        c.text = $text,
                        c.source = $filename
                    MERGE (img)-[:HAS_CHUNK]->(c);
                """, filename=analysis["filename"], c_id=ch["id"], text=ch["text"])

                for ent in ch.get("entities", []):
                    session.run("""
                        MATCH (c:Chunk {id: $c_id})
                        MERGE (e:Entity {name: $ent})
                        MERGE (c)-[:MENTIONS]->(e);
                    """, c_id=ch["id"], ent=ent)

        driver.close()
        print("✅ Image nodes & relationships created in Neo4j!")
    except Exception as e:
        print(f"ℹ️ Neo4j is offline ({e}). Skipping graph push.")


# ======================================================================================
# 5. Direct Visual Q&A
# ======================================================================================

def ask_image_question(image_path: str, question: str) -> str:
    """Directly ask a question about an image using Vision AI."""
    filename = os.path.basename(image_path)
    b64_data, mime_type, _, _ = encode_image_to_base64(image_path)

    print(f"\n🔍 Inspecting '{filename}' to answer: '{question}'...")
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze this image and answer the following question clearly:\n\n{question}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}}
            ]
        }],
        temperature=0.2,
    )
    answer = response.choices[0].message.content or ""
    usage = response.usage

    print("\n" + "=" * 65)
    print("                VISION AI ANSWER                ")
    print("=" * 65)
    print(answer)
    print("=" * 65)
    if usage:
        print(f"🔥 Tokens: Input={usage.prompt_tokens}, Output={usage.completion_tokens}, Total={usage.total_tokens}")
    return answer


# ======================================================================================
# 6. Pipeline Runner
# ======================================================================================

def process_image_file(image_path: str, export_json: Optional[str] = None):
    """End-to-end image processing: Analyze -> Chunk (1500/250) -> Embed -> Save to DB."""
    print("=" * 70)
    print(" 🖼️  STANDALONE MULTIMODAL IMAGE INGESTION PIPELINE")
    print("=" * 70)

    # 1. Vision AI Analysis
    analysis = analyze_image_with_vision(image_path)
    print(f"✅ Image Category: {analysis['category']}")
    print(f"✅ Entities Found: {', '.join(analysis['entities']) if analysis['entities'] else 'None'}")
    print(f"🔥 Vision AI Tokens Burned: {analysis['tokens']['total']} tokens")

    # 2. Chunking (1500 / 250)
    chunks = chunk_image_analysis(analysis, chunk_size=1500, overlap=250)
    print(f"✅ Created {len(chunks)} searchable chunk(s) (1500/250).")

    # 3. Ingest into PostgreSQL pgvector
    ingest_image_chunks_to_postgres(chunks)

    # 4. Ingest into Neo4j
    ingest_image_to_neo4j(analysis, chunks)

    # 5. Optional JSON export
    out_file = export_json or f"{os.path.splitext(analysis['filename'])[0]}_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"analysis": analysis, "chunks": chunks}, f, indent=2)
    print(f"💾 Full analysis saved to '{out_file}'")
    print("=" * 70)
    print(" 🎉 Image Ingestion Completed! You can now query it in chatbot.py.")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--ask" and len(sys.argv) >= 4:
            img = sys.argv[2]
            q = " ".join(sys.argv[3:])
            ask_image_question(img, q)
        else:
            target_image = sys.argv[1]
            process_image_file(target_image)
    else:
        # Default self-test on an extracted diagram if available
        media_files = [os.path.join(MEDIA_DIR, f) for f in os.listdir(MEDIA_DIR) if f.endswith((".png", ".jpg"))] if os.path.exists(MEDIA_DIR) else []
        if media_files:
            process_image_file(media_files[0])
        else:
            print("Usage:")
            print("  python image_handler.py <path_to_image.png>")
            print("  python image_handler.py --ask <path_to_image.png> \"What is this diagram showing?\"")
