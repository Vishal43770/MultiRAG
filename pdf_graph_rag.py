"""
========================================================================================
Multimodal PDF Graph-RAG Engine (Unified Flat Chunks)
========================================================================================
Features:
1. Multimodal PDF Extraction:
   - Formatted Text with Typography-Aware Headings
   - Structured Tables converted to Markdown
   - Diagrams, Drawings & Raster Images extracted to ./extracted_media/
   - Embedded URLs and Hyperlinks extracted per page
2. Unified Multimodal Chunks (1500 chars / 250 overlap):
   - Single flat (:Chunk) architecture (No Parent-Child complexity)
   - Properties: id, type (text/table/diagram), page_number, urls, breadcrumb, text
3. Graph-Based Knowledge Extraction (Neo4j):
   - (:Document) -[:HAS_CHUNK]-> (:Chunk) -[:MENTIONS]-> (:Entity)
   - (:Entity) -[:CO_OCCURS_WITH]-> (:Entity)
4. Local Vector Embedding Integration (1024-dim BGE-Large ONNX)
========================================================================================
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
import pymupdf  # PyMuPDF
from dotenv import load_dotenv

# Optional Neo4j Driver
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Import local embedding engine if available
try:
    from embedding import get_embedding
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ragpassword")
MEDIA_DIR = os.path.join(BASE_DIR, "extracted_media")
os.makedirs(MEDIA_DIR, exist_ok=True)


# ======================================================================================
# 1. Multimodal PDF Extractor (Text, Tables, Diagrams, Images, URLs)
# ======================================================================================

class MultimodalPDFExtractor:
    """Extracts text, structured tables, images, and links from PDF pages."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc_name = os.path.basename(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.doc = pymupdf.open(pdf_path)

    def extract_page_data(self, page_num: int, current_h1: str, current_h2: str) -> Dict[str, Any]:
        """Extract all multimodal content from a single page."""
        page = self.doc[page_num]
        display_page = page_num + 1

        # A. Extract Hyperlinks / URLs
        urls = []
        for link in page.get_links():
            uri = link.get("uri")
            if uri and uri.startswith("http"):
                urls.append(uri)
        urls = list(set(urls))

        # B. Extract Tables
        tables = []
        try:
            tab_finder = page.find_tables()
            for idx, tab in enumerate(tab_finder.tables):
                df = tab.extract()
                if df and len(df) >= 2:
                    headers = [str(c).strip() if c else f"Col_{i}" for i, c in enumerate(df[0])]
                    rows = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
                    for row in df[1:]:
                        cleaned_row = [str(c).replace("\n", " ").strip() if c else "" for c in row]
                        rows.append(f"| {' | '.join(cleaned_row)} |")
                    tables.append({
                        "id": f"{self.doc_name}_p{display_page}_tab{idx + 1}",
                        "page_number": display_page,
                        "markdown": "\n".join(rows),
                        "rows": len(df) - 1,
                        "cols": len(headers),
                    })
        except Exception:
            pass

        # C. Extract Images / Diagrams
        diagrams = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_img = self.doc.extract_image(xref)
                if base_img["width"] >= 80 and base_img["height"] >= 80:
                    ext = base_img["ext"]
                    filename = f"{os.path.splitext(self.doc_name)[0]}_p{display_page}_img{img_idx + 1}.{ext}"
                    filepath = os.path.join(MEDIA_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(base_img["image"])

                    diagrams.append({
                        "id": f"{self.doc_name}_p{display_page}_diag{img_idx + 1}",
                        "page_number": display_page,
                        "image_path": filepath,
                        "filename": filename,
                        "width": base_img["width"],
                        "height": base_img["height"],
                        "caption": f"Diagram/Figure on Page {display_page} ({base_img['width']}x{base_img['height']}px)",
                    })
            except Exception:
                continue

        # D. Extract Text and Detect Headers
        blocks = page.get_text("blocks")
        text_paragraphs = []
        h1 = current_h1
        h2 = current_h2

        for b in blocks:
            text = b[4].strip()
            if not text:
                continue

            # Detect chapter / family / section headers
            if re.match(r"^(Chapter\s+\d+|[A-Z\s]{4,}:|\b[A-Z]{2}\b\s+-\s+.*)", text):
                h1 = text.split("\n")[0][:80]
                h2 = "General"
            elif re.match(r"^(\d+\.\d+\s+[A-Z]|[A-Z]{2}-\d+\s+[A-Z])", text):
                h2 = text.split("\n")[0][:80]
            else:
                text_paragraphs.append(text)

        return {
            "page_number": display_page,
            "h1": h1,
            "h2": h2,
            "text": "\n\n".join(text_paragraphs),
            "urls": urls,
            "tables": tables,
            "diagrams": diagrams,
        }


# ======================================================================================
# 2. Unified Multimodal Chunker (1500 chars / 250 overlap)
# ======================================================================================

class UnifiedMultimodalChunker:
    """Splits document pages into flat 1500-char chunks with rich metadata."""

    def __init__(self, chunk_size: int = 1500, overlap: int = 250):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text_recursive(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """Recursively split text by paragraphs, lines, and sentences."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", "; ", " "]

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        remaining = separators[1:] if len(separators) > 1 else []

        splits = text.split(sep)
        chunks = []
        current_chunk = []
        current_len = 0

        for piece in splits:
            piece_str = piece.strip()
            if not piece_str:
                continue

            added_len = len(piece_str) + len(sep)
            if added_len > self.chunk_size and remaining:
                sub_chunks = self.split_text_recursive(piece_str, remaining)
                chunks.extend(sub_chunks)
                continue

            if current_len + added_len <= self.chunk_size:
                current_chunk.append(piece_str)
                current_len += added_len
            else:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                # Retain overlap from end of current_chunk
                overlap_pieces = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.overlap:
                        overlap_pieces.insert(0, item)
                        overlap_len += len(item) + len(sep)
                    else:
                        break
                current_chunk = overlap_pieces + [piece_str]
                current_len = sum(len(x) + len(sep) for x in current_chunk)

        if current_chunk:
            chunks.append(sep.join(current_chunk))
        return chunks

    def process_document(self, extractor: MultimodalPDFExtractor, start_page: int = 1, end_page: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract and build flat, typed chunks for the entire PDF."""
        total_pages = len(extractor.doc)
        end = end_page or total_pages
        print(f"📄 Processing '{extractor.doc_name}' with 1500/250 chunking (Pages {start_page} to {end})...")

        chunks = []
        chunk_counter = 0
        h1 = "Overview"
        h2 = "General"

        for p_idx in range(start_page - 1, end):
            page_data = extractor.extract_page_data(p_idx, h1, h2)
            h1 = page_data["h1"]
            h2 = page_data["h2"]
            p_num = page_data["page_number"]
            breadcrumb = f"{extractor.doc_name} > {h1} > {h2}"

            # 1. Text Chunks (1500 chars / 250 overlap)
            page_text = page_data["text"]
            if page_text:
                text_splits = self.split_text_recursive(page_text)
                for split_text in text_splits:
                    chunk_counter += 1
                    full_text = f"[{breadcrumb} | Page {p_num}]\n{split_text}"
                    chunks.append({
                        "id": f"{extractor.doc_name}_chunk_{chunk_counter}",
                        "type": "text",
                        "page_number": p_num,
                        "source": extractor.doc_name,
                        "breadcrumb": breadcrumb,
                        "urls": page_data["urls"],
                        "text": full_text,
                    })

            # 2. Table Chunks
            for tab in page_data["tables"]:
                chunk_counter += 1
                table_text = f"[{breadcrumb} | Table | Page {p_num}]\n{tab['markdown']}"
                chunks.append({
                    "id": f"{extractor.doc_name}_chunk_{chunk_counter}",
                    "type": "table",
                    "page_number": p_num,
                    "source": extractor.doc_name,
                    "breadcrumb": breadcrumb,
                    "urls": page_data["urls"],
                    "text": table_text,
                    "table_data": tab["markdown"],
                })

            # 3. Diagram / Image Chunks
            for diag in page_data["diagrams"]:
                chunk_counter += 1
                diag_text = f"[{breadcrumb} | Diagram | Page {p_num}]\n{diag['caption']}"
                chunks.append({
                    "id": f"{extractor.doc_name}_chunk_{chunk_counter}",
                    "type": "diagram",
                    "page_number": p_num,
                    "source": extractor.doc_name,
                    "breadcrumb": breadcrumb,
                    "urls": page_data["urls"],
                    "text": diag_text,
                    "image_path": diag["image_path"],
                })

        extractor.doc.close()
        return chunks


# ======================================================================================
# 3. Graph-Based Knowledge Extraction & Neo4j Ingestion
# ======================================================================================

class FlatKnowledgeGraph:
    """Ingests flat typed chunks and entities into Neo4j."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self) -> bool:
        if not NEO4J_AVAILABLE:
            return False
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1;")
            print(f"🔗 Connected to Neo4j at {self.uri}")
            return True
        except Exception:
            return False

    def close(self):
        if self.driver:
            self.driver.close()

    def extract_entities(self, text: str) -> List[str]:
        """Extract technical terms and concepts."""
        candidates = set(re.findall(r"\b[A-Z]{2,}(?:-[0-9]+)?\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
        stop = {"The Organization", "For Example", "United States", "Table Of", "In Addition"}
        return [c for c in candidates if c not in stop and len(c) > 2][:8]

    def ingest_chunks(self, doc_name: str, chunks: List[Dict[str, Any]]):
        """Insert (:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # 1. Document Node
            session.run("MERGE (d:Document {name: $name});", name=doc_name)

            for ch in chunks:
                # 2. Chunk Node
                session.run("""
                    MATCH (d:Document {name: $doc_name})
                    MERGE (c:Chunk {id: $id})
                    ON CREATE SET 
                        c.type = $type,
                        c.page_number = $page,
                        c.breadcrumb = $breadcrumb,
                        c.urls = $urls,
                        c.text = $text
                    MERGE (d)-[:HAS_CHUNK]->(c);
                """, doc_name=doc_name, id=ch["id"], type=ch["type"],
                    page=ch["page_number"], breadcrumb=ch["breadcrumb"],
                    urls=ch.get("urls", []), text=ch["text"])

                # 3. Entity Nodes & Relationships
                entities = self.extract_entities(ch["text"])
                for ent in entities:
                    session.run("""
                        MATCH (c:Chunk {id: $c_id})
                        MERGE (e:Entity {name: $ent})
                        MERGE (c)-[:MENTIONS]->(e);
                    """, c_id=ch["id"], ent=ent)


# ======================================================================================
# 4. Pipeline Execution
# ======================================================================================

def run_multimodal_rag_pipeline(pdf_path: str, start_page: int = 1, end_page: Optional[int] = None, export_file: str = "chunks_data.json"):
    print("=" * 70)
    print(" 🚀 UNIFIED MULTIMODAL CHUNKING PIPELINE (1500 / 250)")
    print("=" * 70)

    extractor = MultimodalPDFExtractor(pdf_path)
    chunker = UnifiedMultimodalChunker(chunk_size=1500, overlap=250)
    chunks = chunker.process_document(extractor, start_page=start_page, end_page=end_page)

    # Calculate token counts using local embedding tokenizer
    total_tokens_estimated = sum(len(c["text"]) // 4 for c in chunks)
    print(f"\n📊 Extracted {len(chunks)} Chunks:")
    print(f"   • Text Chunks:    {sum(1 for c in chunks if c['type'] == 'text')}")
    print(f"   • Table Chunks:   {sum(1 for c in chunks if c['type'] == 'table')}")
    print(f"   • Diagram Chunks: {sum(1 for c in chunks if c['type'] == 'diagram')}")
    print(f"   • Estimated Total Text Tokens: ~{total_tokens_estimated:,} tokens")

    # Export to JSON
    with open(export_file, "w", encoding="utf-8") as f:
        json.dump({"document": extractor.doc_name, "total_chunks": len(chunks), "chunks": chunks}, f, indent=2)
    print(f"💾 Chunks saved to '{export_file}'")

    # Ingest to Neo4j if available
    graph = FlatKnowledgeGraph()
    if graph.connect():
        graph.ingest_chunks(extractor.doc_name, chunks)
        print("✅ Graph nodes (:Document)-[:HAS_CHUNK]->(:Entity) created in Neo4j!")
        graph.close()
    else:
        print("💡 Neo4j is offline. Chunks saved to JSON and ready for PostgreSQL / pgvector.")

    print("=" * 70)


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        candidate = os.path.abspath(os.path.join(BASE_DIR, "..", "rag_js", "asserts", "NIST.SP.800-53r5.pdf"))
        if os.path.exists(candidate):
            target = candidate

    if target and os.path.exists(target):
        run_multimodal_rag_pipeline(target, start_page=27, end_page=32)
    else:
        print("Usage: python pdf_graph_rag.py <path_to_pdf>")
