"""
PaperGraph FastAPI Application
- Ingest ArXiv papers → Neo4j
- RAG over knowledge graph
- Session-isolated DB (cleared per session)
- Interactive graph visualization
"""

import os
import re
import time
import json
import uuid
import asyncio
import tempfile
import contextvars
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Optional, Any
from collections import defaultdict

import arxiv
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
from typing import List, Optional as Opt

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
NEO4J_URI       = os.getenv("NEO4J_URI")
NEO4J_AUTH      = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE")
HF_TOKEN        = os.getenv("HF_TOKEN")
HF_EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
EMBED_DIM       = 768
MAX_PAPERS      = 3
MAX_RETRIES     = 2
TOP_K_SECTIONS  = 5
LLM_MODEL_NAME  = "gemini-2.5-flash"

driver     = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
hf_client  = InferenceClient(token=HF_TOKEN)
llm        = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)

# ── Session store ──────────────────────────────────────────────────────────────
sessions: dict[str, dict] = {}   # session_id → {status, papers, agent, thread_id, messages}
OBS_CONTEXT: contextvars.ContextVar[Optional[dict[str, str]]] = contextvars.ContextVar("obs_context", default=None)


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MODEL_INPUT_COST_PER_1M_USD = _env_float("MODEL_INPUT_COST_PER_1M_USD", 0.0)
MODEL_OUTPUT_COST_PER_1M_USD = _env_float("MODEL_OUTPUT_COST_PER_1M_USD", 0.0)
CHAT_TIMEOUT_SECONDS = _env_int("CHAT_TIMEOUT_SECONDS", 180)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _truncate_text(value: str, max_len: int = 320) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= max_len else text[:max_len] + "…"


def _ensure_observations(session_obj: dict) -> dict:
    obs = session_obj.get("observations")
    if obs:
        return obs
    obs = {
        "meta": {
            "model": LLM_MODEL_NAME,
            "langsmith_enabled": bool(os.getenv("LANGSMITH_API_KEY")),
            "langsmith_project": os.getenv("LANGSMITH_PROJECT", "default"),
            "cost_model": {
                "input_per_1m_usd": MODEL_INPUT_COST_PER_1M_USD,
                "output_per_1m_usd": MODEL_OUTPUT_COST_PER_1M_USD,
            },
        },
        "ingestion_events": [],
        "chat_turns": [],
        "totals": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    }
    session_obj["observations"] = obs
    return obs


def _append_ingestion_event(
    session_obj: dict,
    stage: str,
    status: str = "ok",
    latency_ms: Optional[float] = None,
    details: Optional[dict] = None,
):
    obs = _ensure_observations(session_obj)
    event: dict[str, Any] = {
        "timestamp": _now_iso(),
        "stage": stage,
        "status": status,
    }
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
    if details:
        event["details"] = details
    obs["ingestion_events"].append(event)
    if len(obs["ingestion_events"]) > 400:
        obs["ingestion_events"] = obs["ingestion_events"][-400:]


def _get_chat_turn(obs: dict, turn_id: str) -> Optional[dict]:
    for turn in reversed(obs.get("chat_turns", [])):
        if turn.get("turn_id") == turn_id:
            return turn
    return None


def _start_chat_turn(session_obj: dict, query: str) -> str:
    obs = _ensure_observations(session_obj)
    turn_id = f"turn_{len(obs['chat_turns']) + 1}"
    obs["chat_turns"].append({
        "turn_id": turn_id,
        "started_at": _now_iso(),
        "status": "running",
        "query": query,
        "answer": "",
        "latency_ms": None,
        "node_events": [],
        "llm_calls": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "estimated_cost_usd": 0.0,
    })
    return turn_id


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    in_cost = (input_tokens / 1_000_000) * MODEL_INPUT_COST_PER_1M_USD
    out_cost = (output_tokens / 1_000_000) * MODEL_OUTPUT_COST_PER_1M_USD
    return round(in_cost + out_cost, 8)


def _extract_usage_metadata(response) -> dict:
    usage = {}
    raw_usage = getattr(response, "usage_metadata", None)
    if isinstance(raw_usage, dict):
        usage = raw_usage
    elif isinstance(getattr(response, "response_metadata", None), dict):
        meta = response.response_metadata
        usage = (
            meta.get("token_usage")
            or meta.get("usage_metadata")
            or meta.get("usage")
            or {}
        )

    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_token_count")
        or usage.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("candidates_token_count")
        or usage.get("completion_tokens")
        or 0
    )
    total_tokens = int(
        usage.get("total_tokens")
        or usage.get("total_token_count")
        or (input_tokens + output_tokens)
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _record_node_event(session_id: str, turn_id: str, event: dict):
    session_obj = sessions.get(session_id)
    if not session_obj:
        return
    obs = _ensure_observations(session_obj)
    turn = _get_chat_turn(obs, turn_id)
    if not turn:
        return
    turn["node_events"].append(event)


def _record_llm_event(session_id: str, turn_id: str, event: dict):
    session_obj = sessions.get(session_id)
    if not session_obj:
        return
    obs = _ensure_observations(session_obj)
    turn = _get_chat_turn(obs, turn_id)
    if not turn:
        return

    turn["llm_calls"].append(event)
    if len(turn["llm_calls"]) > 80:
        turn["llm_calls"] = turn["llm_calls"][-80:]

    if event.get("status") == "ok":
        usage = event.get("token_usage", {})
        in_tokens = int(usage.get("input_tokens", 0))
        out_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", in_tokens + out_tokens))
        cost = float(event.get("estimated_cost_usd", 0.0))

        turn_usage = turn["token_usage"]
        turn_usage["input_tokens"] += in_tokens
        turn_usage["output_tokens"] += out_tokens
        turn_usage["total_tokens"] += total_tokens
        turn["estimated_cost_usd"] = round(turn.get("estimated_cost_usd", 0.0) + cost, 8)

        totals = obs["totals"]
        totals["input_tokens"] += in_tokens
        totals["output_tokens"] += out_tokens
        totals["total_tokens"] += total_tokens
        totals["estimated_cost_usd"] = round(totals.get("estimated_cost_usd", 0.0) + cost, 8)


def _complete_chat_turn(
    session_id: str,
    turn_id: str,
    status: str,
    answer: Optional[str] = None,
    error: Optional[str] = None,
    latency_ms: Optional[float] = None,
):
    session_obj = sessions.get(session_id)
    if not session_obj:
        return
    obs = _ensure_observations(session_obj)
    turn = _get_chat_turn(obs, turn_id)
    if not turn:
        return

    turn["status"] = status
    turn["finished_at"] = _now_iso()
    if latency_ms is not None:
        turn["latency_ms"] = latency_ms
    if answer is not None:
        turn["answer"] = answer
    if error:
        turn["error"] = _truncate_text(error, 500)


def _message_preview(messages: list) -> str:
    previews = []
    for msg in messages[-2:]:
        role = getattr(msg, "type", msg.__class__.__name__)
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = json.dumps(content)
        previews.append(f"{role}: {_truncate_text(str(content), 180)}")
    return " | ".join(previews)


def _response_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(item if isinstance(item, str) else str(item) for item in content)
    return str(content or "")

app = FastAPI(title="PaperGraph")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=TEMPLATES_DIR), name="static")

# ══════════════════════════════════════════════════════════════════════════════
#  NEO4J SCHEMA SETUP
# ══════════════════════════════════════════════════════════════════════════════

SETUP_QUERIES = [
    "CREATE CONSTRAINT paper_arxiv_id  IF NOT EXISTS FOR (p:Paper)   REQUIRE p.arxiv_id   IS UNIQUE",
    "CREATE CONSTRAINT author_id       IF NOT EXISTS FOR (a:Author)  REQUIRE a.author_id  IS UNIQUE",
    "CREATE CONSTRAINT section_id      IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
    "CREATE CONSTRAINT concept_name    IF NOT EXISTS FOR (c:Concept) REQUIRE c.name_norm  IS UNIQUE",
    "CREATE CONSTRAINT method_name     IF NOT EXISTS FOR (m:Method)  REQUIRE m.name_norm  IS UNIQUE",
    "CREATE CONSTRAINT dataset_name    IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name_norm  IS UNIQUE",
    "CREATE CONSTRAINT topic_name      IF NOT EXISTS FOR (t:Topic)   REQUIRE t.name_norm  IS UNIQUE",
    (
        "CREATE VECTOR INDEX section_embedding IF NOT EXISTS "
        "FOR (s:Section) ON s.embedding "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine' }} }}"
    ),
    "CREATE INDEX paper_year  IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    "CREATE INDEX paper_stub  IF NOT EXISTS FOR (p:Paper) ON (p.stub)",
]


def setup_schema():
    for q in SETUP_QUERIES:
        try:
            driver.execute_query(q, database_=NEO4J_DATABASE)
        except Exception:
            pass


def clear_database():
    """Wipe ALL nodes and relationships."""
    driver.execute_query("MATCH (n) DETACH DELETE n", database_=NEO4J_DATABASE)


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS FOR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class ExtractedEntity(BaseModel):
    name: str
    description: Opt[str] = None
    confidence: float = 1.0

class TopicEntity(BaseModel):
    name: str
    description: Opt[str] = None
    parent_topic: Opt[str] = None
    confidence: float = 1.0

class ConceptPair(BaseModel):
    concept_a: str
    concept_b: str
    confidence: float = 1.0

class MethodExtension(BaseModel):
    child_method: str
    parent_method: str
    confidence: float = 1.0

class PaperExtraction(BaseModel):
    paper_summary: str = ""
    concepts_introduced: List[ExtractedEntity] = Field(default_factory=list)
    concepts_applied: List[ExtractedEntity] = Field(default_factory=list)
    methods: List[ExtractedEntity] = Field(default_factory=list)
    datasets: List[ExtractedEntity] = Field(default_factory=list)
    topics: List[TopicEntity] = Field(default_factory=list)
    concept_relations: List[ConceptPair] = Field(default_factory=list)
    method_extensions: List[MethodExtension] = Field(default_factory=list)
    extends_papers: List[str] = Field(default_factory=list)
    contradicts_papers: List[str] = Field(default_factory=list)

structured_llm = llm.with_structured_output(PaperExtraction)

# ══════════════════════════════════════════════════════════════════════════════
#  INGESTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CONTENT_TYPES = {"NarrativeText", "Text", "UncategorizedText", "ListItem", "FigureCaption", "Table"}
_REF_HEADERS   = {"references", "bibliography", "reference list", "works cited"}
_ARXIV_PATS    = [
    r"arXiv[:\s]+(\d{4}\.\d{4,5})(?:v\d+)?",
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",
    r"\[(\d{4}\.\d{4,5})\]",
]

def _arxiv_id(text: str):
    for pat in _ARXIV_PATS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return re.sub(r"v\d+$", "", m.group(1))
    return None

def _title_from_entry(text: str):
    m = re.search(r'[""\u201c\u201e]([^""\u201c\u201d\u201e]{15,180})[""\u201d\u201e]', text)
    if m: return m.group(1).strip()
    m = re.search(r'\(\d{4}\)[.,]?\s+([A-Z][^\n]{20,180}?)(?:\.\s+[A-Z]|\.$|,\s+In\b|,\s+Proc)', text)
    if m: return m.group(1).strip()
    return None

def partition_pdf_elements(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return partition_pdf(filename=tmp_path, strategy="fast", infer_table_structure=False)
    finally:
        os.unlink(tmp_path)

def parse_sections(pdf_bytes: bytes, max_sections: int = 15):
    elements = partition_pdf_elements(pdf_bytes)
    sections, current_title, current_parts, order, in_references = [], None, [], 0, False

    def flush():
        nonlocal order
        if current_title and current_parts and not in_references:
            content = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
            if len(content) > 100:
                sections.append({"title": current_title, "content": content[:3000], "order": order})
                order += 1

    ref_elements = []
    for el in elements:
        el_type = type(el).__name__
        el_text = (el.text or "").strip()
        if not el_text:
            continue
        if el_type == "Title":
            if el_text.lower().strip(".:") in _REF_HEADERS:
                flush(); in_references = True; continue
            if not in_references:
                flush(); current_title = el_text; current_parts = []
        elif in_references:
            ref_elements.append(el_text)
        elif el_type in _CONTENT_TYPES:
            if current_title is None:
                current_title = "Introduction"
            current_parts.append(el_text)

    if not in_references:
        flush()

    if not sections:
        all_words = " ".join((el.text or "") for el in elements if type(el).__name__ in _CONTENT_TYPES).split()
        chunk_sz = 500
        sections = [
            {"title": f"Chunk {i//chunk_sz+1}", "content": " ".join(all_words[i:i+chunk_sz]), "order": i//chunk_sz}
            for i in range(0, min(len(all_words), chunk_sz*max_sections), chunk_sz)
            if " ".join(all_words[i:i+chunk_sz]).strip()
        ]

    cited_refs, seen_arxiv, seen_titles = [], set(), set()
    for entry in ref_elements:
        if len(entry) < 20: continue
        aid = _arxiv_id(entry)
        if aid:
            if aid not in seen_arxiv:
                seen_arxiv.add(aid)
                cited_refs.append({"arxiv_id": aid, "title": _title_from_entry(entry)})
        else:
            t = _title_from_entry(entry)
            if t:
                tnorm = t.lower().strip()
                if tnorm not in seen_titles and len(tnorm.split()) >= 4:
                    seen_titles.add(tnorm)
                    cited_refs.append({"arxiv_id": None, "title": t})

    return sections[:max_sections], cited_refs


EXTRACTION_PROMPT = """You are a scientific paper analyst. Extract research entities from the paper below.

Title: {title}
Abstract: {abstract}

Extract ALL of the following. For every entity assign a confidence score (0.0-1.0):

- paper_summary      : 2-3 sentence plain-English summary
- concepts_introduced: NEW concepts this paper PROPOSES
- concepts_applied   : EXISTING concepts this paper USES
- methods            : Specific algorithms, models, or techniques
- datasets           : Datasets used for training or evaluation
- topics             : Broad research areas with parent_topic chain
- concept_relations  : Pairs of closely related concepts (RELATED_TO)
- method_extensions  : Method hierarchy pairs (child EXTENDS parent)
- extends_papers     : Exact titles of prior papers this builds upon
- contradicts_papers : Exact titles of prior papers this disputes

Keep entity names concise (3-6 words). Be conservative with confidence scores."""


def extract_entities(paper: dict) -> PaperExtraction:
    prompt = EXTRACTION_PROMPT.format(title=paper["title"], abstract=paper["abstract"])
    return structured_llm.invoke(prompt)


def _embed(text: str) -> list:
    arr = np.array(hf_client.feature_extraction(text, model=HF_EMBED_MODEL))
    if arr.ndim == 3: arr = arr[0]
    if arr.ndim == 2: arr = arr.mean(axis=0)
    return arr.tolist()


def ingest_paper(paper: dict, extraction: PaperExtraction, sections: list, cited_refs: list):
    aid = paper["arxiv_id"]

    def items(entities): return [e.model_dump() for e in entities]

    # 1. Paper node
    driver.execute_query("""
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title, p.title_norm = toLower(trim($title)),
            p.abstract = $abstract, p.summary = $summary,
            p.pdf_url = $pdf_url, p.published_date = date($published),
            p.year = $year, p.category = $category, p.stub = false
    """, arxiv_id=aid, title=paper["title"], abstract=paper["abstract"],
         summary=extraction.paper_summary, pdf_url=paper["pdf_url"],
         published=paper["published"], year=paper["year"],
         category=paper["category"], database_=NEO4J_DATABASE)

    # 2. Authors
    if paper["authors"]:
        driver.execute_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            UNWIND $authors AS author_name
              MERGE (a:Author {name_norm: toLower(trim(author_name))})
              ON CREATE SET a.author_id = randomUUID(), a.name = author_name
              MERGE (p)-[:AUTHORED_BY]->(a)
        """, arxiv_id=aid, authors=paper["authors"], database_=NEO4J_DATABASE)
        if len(paper["authors"]) > 1:
            driver.execute_query("""
                UNWIND $authors AS name_a UNWIND $authors AS name_b
                WITH name_a, name_b WHERE name_a < name_b
                MATCH (a:Author {name_norm: toLower(trim(name_a))})
                MATCH (b:Author {name_norm: toLower(trim(name_b))})
                MERGE (a)-[r:CO_AUTHORED]-(b)
                ON CREATE SET r.paper_count = 1
                ON MATCH  SET r.paper_count = r.paper_count + 1
            """, authors=paper["authors"], database_=NEO4J_DATABASE)

    # 3-4. Concepts
    for rel, entities in [("INTRODUCES", extraction.concepts_introduced), ("APPLIES", extraction.concepts_applied)]:
        if entities:
            driver.execute_query(f"""
                MATCH (p:Paper {{arxiv_id: $arxiv_id}})
                UNWIND $items AS item
                  MERGE (c:Concept {{name_norm: toLower(trim(item.name))}})
                  ON CREATE SET c.concept_id = randomUUID(), c.name = item.name, c.description = item.description
                  MERGE (p)-[r:{rel}]->(c)
                  SET r.source = 'llm', r.confidence = item.confidence
            """, arxiv_id=aid, items=items(entities), database_=NEO4J_DATABASE)

    # 5. Methods
    if extraction.methods:
        driver.execute_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            UNWIND $items AS item
              MERGE (m:Method {name_norm: toLower(trim(item.name))})
              ON CREATE SET m.method_id = randomUUID(), m.name = item.name, m.description = item.description
              MERGE (p)-[r:USES_METHOD]->(m)
              SET r.source = 'llm', r.confidence = item.confidence
        """, arxiv_id=aid, items=items(extraction.methods), database_=NEO4J_DATABASE)

    # 6. Datasets
    if extraction.datasets:
        driver.execute_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            UNWIND $items AS item
              MERGE (d:Dataset {name_norm: toLower(trim(item.name))})
              ON CREATE SET d.dataset_id = randomUUID(), d.name = item.name, d.description = item.description
              MERGE (p)-[r:USES_DATASET]->(d)
              SET r.source = 'llm', r.confidence = item.confidence
        """, arxiv_id=aid, items=items(extraction.datasets), database_=NEO4J_DATABASE)

    # 7. Topics
    if extraction.topics:
        topic_dicts = [t.model_dump() for t in extraction.topics]
        driver.execute_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            UNWIND $topics AS t
              MERGE (tn:Topic {name_norm: toLower(trim(t.name))})
              ON CREATE SET tn.topic_id = randomUUID(), tn.name = t.name, tn.description = t.description
              MERGE (p)-[r:BELONGS_TO]->(tn)
              SET r.source = 'llm', r.confidence = t.confidence
        """, arxiv_id=aid, topics=topic_dicts, database_=NEO4J_DATABASE)
        driver.execute_query("""
            UNWIND $topics AS t
              WITH t WHERE t.parent_topic IS NOT NULL AND t.parent_topic <> ''
              MERGE (child:Topic  {name_norm: toLower(trim(t.name))})
              MERGE (parent:Topic {name_norm: toLower(trim(t.parent_topic))})
              ON CREATE SET parent.topic_id = randomUUID(), parent.name = t.parent_topic
              MERGE (child)-[:BELONGS_TO]->(parent)
        """, topics=topic_dicts, database_=NEO4J_DATABASE)

    # 8. Concept relations
    if extraction.concept_relations:
        driver.execute_query("""
            UNWIND $pairs AS pair
              MERGE (a:Concept {name_norm: toLower(trim(pair.concept_a))})
              ON CREATE SET a.concept_id = randomUUID(), a.name = pair.concept_a
              MERGE (b:Concept {name_norm: toLower(trim(pair.concept_b))})
              ON CREATE SET b.concept_id = randomUUID(), b.name = pair.concept_b
              MERGE (a)-[r:RELATED_TO]->(b)
              SET r.source = 'llm', r.confidence = pair.confidence
        """, pairs=[p.model_dump() for p in extraction.concept_relations], database_=NEO4J_DATABASE)

    # 9. Method extensions
    if extraction.method_extensions:
        driver.execute_query("""
            UNWIND $exts AS ext
              MERGE (child:Method  {name_norm: toLower(trim(ext.child_method))})
              ON CREATE SET child.method_id = randomUUID(), child.name = ext.child_method
              MERGE (parent:Method {name_norm: toLower(trim(ext.parent_method))})
              ON CREATE SET parent.method_id = randomUUID(), parent.name = ext.parent_method
              MERGE (child)-[r:EXTENDS]->(parent)
              SET r.source = 'llm', r.confidence = ext.confidence
        """, exts=[e.model_dump() for e in extraction.method_extensions], database_=NEO4J_DATABASE)

    # 10-11. Paper lineage
    for rel, titles in [("EXTENDS", extraction.extends_papers), ("CONTRADICTS", extraction.contradicts_papers)]:
        if titles:
            driver.execute_query(f"""
                MATCH (p:Paper {{arxiv_id: $arxiv_id}})
                UNWIND $titles AS t
                  MERGE (p2:Paper {{title_norm: toLower(trim(t))}})
                  ON CREATE SET p2.title = t, p2.stub = true
                  MERGE (p)-[r:{rel}]->(p2)
                  SET r.source = 'llm'
            """, arxiv_id=aid, titles=titles, database_=NEO4J_DATABASE)

    # 12. Citations
    if cited_refs:
        arxiv_refs = [r for r in cited_refs if r.get("arxiv_id")]
        title_refs = [r for r in cited_refs if not r.get("arxiv_id") and r.get("title")]
        if arxiv_refs:
            driver.execute_query("""
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                UNWIND $refs AS ref
                  MERGE (p2:Paper {arxiv_id: ref.arxiv_id})
                  ON CREATE SET p2.stub = true, p2.title = ref.title
                  MERGE (p)-[:CITES]->(p2)
            """, arxiv_id=aid, refs=arxiv_refs, database_=NEO4J_DATABASE)
        if title_refs:
            driver.execute_query("""
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                UNWIND $refs AS ref
                  MERGE (p2:Paper {title_norm: toLower(trim(ref.title))})
                  ON CREATE SET p2.paper_id = randomUUID(), p2.title = ref.title, p2.stub = true
                  MERGE (p)-[:CITES]->(p2)
            """, arxiv_id=aid, refs=title_refs, database_=NEO4J_DATABASE)

    # 13. Sections + embeddings
    section_ids = []
    for sec in sections:
        section_id = f"{aid}_s{sec['order']}"
        section_ids.append(section_id)
        try:
            embedding = _embed(sec["content"])
        except Exception:
            embedding = None
        driver.execute_query("""
            MERGE (s:Section {section_id: $section_id})
            SET s.title = $title, s.content = $content,
                s.section_order = $order, s.embedding = $embedding
            WITH s
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:HAS_SECTION]->(s)
        """, section_id=section_id, title=sec["title"], content=sec["content"],
             order=sec["order"], arxiv_id=aid, embedding=embedding,
             database_=NEO4J_DATABASE)

    for i in range(len(section_ids) - 1):
        driver.execute_query("""
            MATCH (a:Section {section_id: $sid_a})
            MATCH (b:Section {section_id: $sid_b})
            MERGE (a)-[:NEXT_SECTION]->(b)
        """, sid_a=section_ids[i], sid_b=section_ids[i+1], database_=NEO4J_DATABASE)


# ══════════════════════════════════════════════════════════════════════════════
#  RAG AGENT (LangGraph)
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_CONTEXT = """
Neo4j Graph Schema:

NODE LABELS & KEY PROPERTIES:
  Paper    : arxiv_id, title, abstract, summary, year, category, stub (bool)
  Author   : author_id, name, name_norm
  Section  : section_id, title, content, section_order, embedding (768-dim)
  Concept  : concept_id, name, name_norm, description
  Method   : method_id, name, name_norm, description
  Dataset  : dataset_id, name, name_norm, description
  Topic    : topic_id, name, name_norm, description

RELATIONSHIPS:
  (Paper)-[:AUTHORED_BY]->(Author)
  (Author)-[:CO_AUTHORED]-(Author)
  (Paper)-[:HAS_SECTION]->(Section)
  (Section)-[:NEXT_SECTION]->(Section)
  (Paper)-[:INTRODUCES]->(Concept)
  (Paper)-[:APPLIES]->(Concept)
  (Paper)-[:USES_METHOD]->(Method)
  (Paper)-[:USES_DATASET]->(Dataset)
  (Paper)-[:BELONGS_TO]->(Topic)
  (Topic)-[:BELONGS_TO]->(Topic)
  (Concept)-[:RELATED_TO]->(Concept)
  (Method)-[:EXTENDS]->(Method)
  (Paper)-[:EXTENDS]->(Paper)
  (Paper)-[:CONTRADICTS]->(Paper)
  (Paper)-[:CITES]->(Paper)

IMPORTANT RULES:
  - Use name_norm for matching Concept/Method/Dataset/Topic
  - Use toLower(trim(...)) on user-provided strings
  - stub=false papers are fully ingested
  - Always return p.arxiv_id AS arxiv_id for any Paper node
  - LIMIT results (10 for papers, 20 for entities)
"""


class AgentState(TypedDict):
    query: str
    messages: Annotated[list, lambda x, y: x + y]
    intent: str
    entities: list[str]
    cypher: str
    cypher_feedback: str
    cypher_retries: int
    cypher_valid: bool
    graph_nodes: list[dict]
    paper_ids: list[str]
    result_feedback: str
    result_retries: int
    result_ok: bool
    sections: list[dict]
    context: str
    answer: str


VALID_LABELS = {"Paper", "Author", "Section", "Concept", "Method", "Dataset", "Topic"}
VALID_RELS   = {"AUTHORED_BY", "CO_AUTHORED", "HAS_SECTION", "NEXT_SECTION",
                "INTRODUCES", "APPLIES", "USES_METHOD", "USES_DATASET",
                "BELONGS_TO", "RELATED_TO", "EXTENDS", "CONTRADICTS", "CITES"}


def _summarize_node_output(result: dict) -> dict:
    if not isinstance(result, dict):
        return {"result_type": type(result).__name__}

    summary = {}
    if "intent" in result:
        summary["intent"] = result["intent"]
    if "entities" in result:
        summary["entities"] = result["entities"][:5]
    if "cypher" in result:
        summary["cypher"] = _truncate_text(result["cypher"], 260)
    if "graph_nodes" in result:
        summary["graph_records"] = len(result["graph_nodes"])
    if "paper_ids" in result:
        summary["paper_ids"] = result["paper_ids"][:5]
    if "sections" in result:
        summary["sections"] = len(result["sections"])
    if "context" in result:
        summary["context_chars"] = len(result["context"])
    if "answer" in result:
        summary["answer_preview"] = _truncate_text(result["answer"], 220)
    if result.get("cypher_feedback"):
        summary["cypher_feedback"] = _truncate_text(result["cypher_feedback"], 220)
    if result.get("result_feedback"):
        summary["result_feedback"] = _truncate_text(result["result_feedback"], 220)

    if not summary:
        for key, value in list(result.items())[:4]:
            if isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)}]"
            else:
                summary[key] = _truncate_text(str(value), 120)
    return summary


def _instrument_node(node_name: str, fn):
    def wrapped(state):
        started_at = time.perf_counter()
        obs_ctx = OBS_CONTEXT.get()
        result = None
        err = None
        try:
            result = fn(state)
            return result
        except Exception as exc:
            err = str(exc)
            raise
        finally:
            if obs_ctx and obs_ctx.get("session_id") and obs_ctx.get("turn_id"):
                event = {
                    "timestamp": _now_iso(),
                    "node": node_name,
                    "status": "error" if err else "ok",
                    "latency_ms": _elapsed_ms(started_at),
                }
                if err:
                    event["error"] = _truncate_text(err, 420)
                else:
                    event["output"] = _summarize_node_output(result if isinstance(result, dict) else {})
                _record_node_event(obs_ctx["session_id"], obs_ctx["turn_id"], event)

    return wrapped


def _run_cypher(cypher: str):
    try:
        results, _, _ = driver.execute_query(cypher, database_=NEO4J_DATABASE)
        return [dict(r) for r in results], None
    except Exception as e:
        return [], str(e)


def _run_cypher_raw(cypher: str, **params):
    try:
        results, _, _ = driver.execute_query(cypher, database_=NEO4J_DATABASE, **params)
        return [dict(r) for r in results], None
    except Exception as e:
        return [], str(e)


def _llm_call(messages: list, call_name: str = "llm_call"):
    for attempt in range(3):
        started_at = time.perf_counter()
        try:
            response = llm.invoke(messages)
            obs_ctx = OBS_CONTEXT.get()
            if obs_ctx and obs_ctx.get("session_id") and obs_ctx.get("turn_id"):
                usage = _extract_usage_metadata(response)
                llm_event = {
                    "timestamp": _now_iso(),
                    "call_name": call_name,
                    "status": "ok",
                    "latency_ms": _elapsed_ms(started_at),
                    "token_usage": usage,
                    "estimated_cost_usd": _estimate_cost_usd(usage["input_tokens"], usage["output_tokens"]),
                    "prompt_preview": _message_preview(messages),
                }
                _record_llm_event(obs_ctx["session_id"], obs_ctx["turn_id"], llm_event)
            return response
        except Exception as e:
            obs_ctx = OBS_CONTEXT.get()
            if obs_ctx and obs_ctx.get("session_id") and obs_ctx.get("turn_id"):
                _record_llm_event(obs_ctx["session_id"], obs_ctx["turn_id"], {
                    "timestamp": _now_iso(),
                    "call_name": call_name,
                    "status": "error",
                    "latency_ms": _elapsed_ms(started_at),
                    "error": _truncate_text(str(e), 320),
                    "prompt_preview": _message_preview(messages),
                })
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < 2:
                    time.sleep(60)
                else:
                    raise
            else:
                raise


def _llm_json(prompt: str, call_name: str = "llm_json") -> dict:
    response = _llm_call([HumanMessage(content=prompt)], call_name=call_name)
    text = _response_text(response).strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


def node_query_analyzer(state: AgentState) -> dict:
    prompt = f"""Analyze this research query and return JSON only.

Query: "{state['query']}"

Return this exact JSON structure:
{{
  "intent": "<one of: conceptual | comparative | lineage | factual>",
  "entities": ["<key term 1>", "<key term 2>", ...]
}}

Intent definitions:
- conceptual  : asking what something is or how it works
- comparative : comparing papers, methods, or concepts
- lineage     : asking about citations, extensions, contradictions
- factual     : specific fact lookup (authors, dates, etc.)

Entities: extract specific method, concept, dataset, author, or topic names.
Return maximum 5 entities, normalized to lowercase."""

    result = _llm_json(prompt, call_name="node_query_analyzer")
    return {
        "intent":          result.get("intent", "conceptual"),
        "entities":        result.get("entities", []),
        "cypher_retries":  0,
        "result_retries":  0,
        "cypher_feedback": "",
        "result_feedback": "",
        "messages":        [HumanMessage(content=state["query"])],
    }


def node_cypher_generator(state: AgentState) -> dict:
    feedback_block = ""
    if state.get("cypher_feedback"):
        feedback_block = (
            f"PREVIOUS ATTEMPT FAILED — fix it:\n"
            f"Cypher tried: {state.get('cypher', '')}\n"
            f"Feedback: {state['cypher_feedback']}\n"
        )

    prompt = f"""{SCHEMA_CONTEXT}

{feedback_block}
Generate a Cypher READ query for this research question.

Query: "{state['query']}"
Intent: {state['intent']}
Extracted entities: {state['entities']}

Rules:
- READ only (MATCH/RETURN, no MERGE/CREATE/DELETE)
- Always return p.arxiv_id AS arxiv_id for any Paper node
- Use LIMIT to cap results
- Return clean column names
- Output ONLY the raw Cypher, no explanation, no markdown fences"""

    response = _llm_call([HumanMessage(content=prompt)], call_name="node_cypher_generator")
    cypher = _response_text(response).strip().strip("```").strip()
    if cypher.lower().startswith("cypher"):
        cypher = cypher[6:].strip()
    return {"cypher": cypher}


def node_cypher_checker(state: AgentState) -> dict:
    cypher = state.get("cypher", "")
    issues = []

    write_kw = re.findall(r"\b(MERGE|CREATE|DELETE|SET|REMOVE|DROP)\b", cypher, re.IGNORECASE)
    if write_kw:
        issues.append(f"Write operations not allowed: {write_kw}")

    used_labels = set(re.findall(r"\([\w]*:(\w+)", cypher))
    bad_labels = used_labels - VALID_LABELS
    if bad_labels:
        issues.append(f"Unknown node labels: {bad_labels}. Valid: {VALID_LABELS}")

    used_rels = set(re.findall(r"\[:(\w+)\]", cypher))
    bad_rels = used_rels - VALID_RELS
    if bad_rels:
        issues.append(f"Unknown relationship types: {bad_rels}. Valid: {VALID_RELS}")

    if "RETURN" not in cypher.upper():
        issues.append("Query must have a RETURN clause.")

    if issues:
        feedback = "Cypher issues found:\n" + "\n".join(f"  - {i}" for i in issues)
        return {
            "cypher_valid":    False,
            "cypher_feedback": feedback,
            "cypher_retries":  state.get("cypher_retries", 0) + 1,
        }
    return {"cypher_valid": True, "cypher_feedback": ""}


def node_graph_retriever(state: AgentState) -> dict:
    records, error = _run_cypher(state["cypher"])
    if error:
        return {
            "graph_nodes":     [],
            "paper_ids":       [],
            "result_ok":       False,
            "result_feedback": f"Neo4j execution error: {error}",
            "result_retries":  state.get("result_retries", 0) + 1,
        }
    paper_ids = list({
        r["arxiv_id"] for r in records
        if r.get("arxiv_id") and isinstance(r["arxiv_id"], str)
    })
    return {"graph_nodes": records, "paper_ids": paper_ids}


def node_result_checker(state: AgentState) -> dict:
    nodes   = state.get("graph_nodes", [])
    retries = state.get("result_retries", 0)

    if state.get("result_feedback", "").startswith("Neo4j execution error"):
        if retries >= MAX_RETRIES:
            return {"result_ok": True, "paper_ids": []}
        return {
            "result_ok":       False,
            "cypher_feedback": state["result_feedback"],
            "cypher_retries":  state.get("cypher_retries", 0) + 1,
        }

    if not nodes:
        if retries >= MAX_RETRIES:
            return {"result_ok": True, "paper_ids": []}
        feedback = (
            "Query returned 0 results. Try: "
            "1) remove strict WHERE filters, "
            "2) use CONTAINS instead of exact match, "
            "3) broaden to related node types."
        )
        return {
            "result_ok":       False,
            "result_feedback": feedback,
            "cypher_feedback": feedback,
            "result_retries":  retries + 1,
            "cypher_retries":  state.get("cypher_retries", 0) + 1,
        }
    return {"result_ok": True, "result_feedback": ""}


def node_vector_retriever(state: AgentState) -> dict:
    query_vec = _embed(state["query"])
    cypher = """
    CALL db.index.vector.queryNodes('section_embedding', $k, $qvec)
    YIELD node AS s, score
    MATCH (p:Paper)-[:HAS_SECTION]->(s)
    WHERE p.arxiv_id IN $paper_ids
    RETURN s.section_id AS section_id, s.title AS section_title,
           s.content AS content, p.arxiv_id AS arxiv_id,
           p.title AS paper_title, score
    """
    records, _ = _run_cypher_raw(cypher, paper_ids=state["paper_ids"], qvec=query_vec, k=TOP_K_SECTIONS * 4)
    return {"sections": records[:TOP_K_SECTIONS]}


def node_global_vector_retriever(state: AgentState) -> dict:
    query_vec = _embed(state["query"])
    cypher = """
    CALL db.index.vector.queryNodes('section_embedding', $k, $qvec)
    YIELD node AS s, score
    MATCH (p:Paper)-[:HAS_SECTION]->(s)
    WHERE p.stub = false
    RETURN s.section_id AS section_id, s.title AS section_title,
           s.content AS content, p.arxiv_id AS arxiv_id,
           p.title AS paper_title, score
    """
    records, _ = _run_cypher_raw(cypher, qvec=query_vec, k=TOP_K_SECTIONS)
    return {"sections": records}


def node_context_builder(state: AgentState) -> dict:
    parts = []
    if state.get("graph_nodes"):
        parts.append("=== Graph Context ===")
        seen = set()
        for node in state["graph_nodes"][:15]:
            line = " | ".join(f"{k}: {v}" for k, v in node.items() if v is not None)
            if line not in seen:
                seen.add(line)
                parts.append(f"  {line}")

    if state.get("sections"):
        parts.append("\n=== Relevant Paper Sections ===")
        for sec in state["sections"]:
            parts.append(
                f"\n[{sec.get('paper_title', 'Unknown')} | {sec.get('arxiv_id', '')}]"
                f" — {sec.get('section_title', '')}\n"
                f"{sec.get('content', '')[:800]}"
            )

    context = "\n".join(parts) if parts else "No relevant context found in the knowledge graph."
    return {"context": context}


def node_answer_generator(state: AgentState) -> dict:
    system = """You are a research assistant with access to a knowledge graph of academic papers.
Answer the user's question using only the provided context.
- Cite papers using their arxiv_id in brackets e.g. [2203.08975]
- Be concise but complete
- If context is insufficient, say so clearly
- Do not hallucinate paper titles or results"""

    user_prompt = f"""Question: {state['query']}

Context:
{state['context']}

Answer:"""

    response = _llm_call(
        [SystemMessage(content=system), HumanMessage(content=user_prompt)],
        call_name="node_answer_generator",
    )
    answer = _response_text(response).strip()
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def route_cypher_checker(state: AgentState) -> Literal["node_cypher_generator", "node_graph_retriever"]:
    if not state.get("cypher_valid", False):
        if state.get("cypher_retries", 0) >= MAX_RETRIES:
            return "node_graph_retriever"
        return "node_cypher_generator"
    return "node_graph_retriever"


def route_result_checker(state: AgentState) -> Literal["node_cypher_generator", "node_vector_retriever", "node_global_vector_retriever"]:
    if not state.get("result_ok", False):
        if state.get("result_retries", 0) < MAX_RETRIES:
            return "node_cypher_generator"
    if state.get("paper_ids"):
        return "node_vector_retriever"
    return "node_global_vector_retriever"


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("node_query_analyzer",          _instrument_node("node_query_analyzer", node_query_analyzer))
    g.add_node("node_cypher_generator",        _instrument_node("node_cypher_generator", node_cypher_generator))
    g.add_node("node_cypher_checker",          _instrument_node("node_cypher_checker", node_cypher_checker))
    g.add_node("node_graph_retriever",         _instrument_node("node_graph_retriever", node_graph_retriever))
    g.add_node("node_result_checker",          _instrument_node("node_result_checker", node_result_checker))
    g.add_node("node_vector_retriever",        _instrument_node("node_vector_retriever", node_vector_retriever))
    g.add_node("node_global_vector_retriever", _instrument_node("node_global_vector_retriever", node_global_vector_retriever))
    g.add_node("node_context_builder",         _instrument_node("node_context_builder", node_context_builder))
    g.add_node("node_answer_generator",        _instrument_node("node_answer_generator", node_answer_generator))

    g.set_entry_point("node_query_analyzer")
    g.add_edge("node_query_analyzer",   "node_cypher_generator")
    g.add_edge("node_cypher_generator", "node_cypher_checker")

    g.add_conditional_edges("node_cypher_checker", route_cypher_checker,
        {"node_cypher_generator": "node_cypher_generator", "node_graph_retriever": "node_graph_retriever"})

    g.add_edge("node_graph_retriever", "node_result_checker")

    g.add_conditional_edges("node_result_checker", route_result_checker, {
        "node_cypher_generator":        "node_cypher_generator",
        "node_vector_retriever":        "node_vector_retriever",
        "node_global_vector_retriever": "node_global_vector_retriever",
    })

    g.add_edge("node_vector_retriever",        "node_context_builder")
    g.add_edge("node_global_vector_retriever", "node_context_builder")
    g.add_edge("node_context_builder",         "node_answer_generator")
    g.add_edge("node_answer_generator",        END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH DATA EXPORT FOR VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def get_graph_data() -> dict:
    """Fetch all nodes + relationships from Neo4j for visualization."""
    # Nodes
    node_query = """
    MATCH (n)
    WHERE NOT n:Section
    RETURN
      id(n)          AS id,
      labels(n)[0]   AS label,
      CASE labels(n)[0]
        WHEN 'Paper'   THEN coalesce(n.title, n.arxiv_id)
        WHEN 'Author'  THEN coalesce(n.name, n.name_norm)
        WHEN 'Concept' THEN coalesce(n.name, n.name_norm)
        WHEN 'Method'  THEN coalesce(n.name, n.name_norm)
        WHEN 'Dataset' THEN coalesce(n.name, n.name_norm)
        WHEN 'Topic'   THEN coalesce(n.name, n.name_norm)
        ELSE toString(id(n))
      END AS display_name,
      CASE labels(n)[0]
        WHEN 'Paper' THEN coalesce(n.arxiv_id, '')
        ELSE ''
      END AS arxiv_id,
      CASE labels(n)[0]
        WHEN 'Paper' THEN coalesce(n.summary, n.abstract, '')
        WHEN 'Concept' THEN coalesce(n.description, '')
        WHEN 'Method'  THEN coalesce(n.description, '')
        WHEN 'Dataset' THEN coalesce(n.description, '')
        WHEN 'Topic'   THEN coalesce(n.description, '')
        ELSE ''
      END AS description,
      CASE WHEN labels(n)[0] = 'Paper' THEN coalesce(n.stub, false) ELSE false END AS is_stub
    LIMIT 300
    """
    # Edges
    edge_query = """
    MATCH (a)-[r]->(b)
    WHERE NOT a:Section AND NOT b:Section
    RETURN
      id(a) AS source,
      id(b) AS target,
      type(r) AS rel_type,
      CASE WHEN r.confidence IS NOT NULL THEN r.confidence ELSE 1.0 END AS weight
    LIMIT 500
    """
    nodes_raw, _ = _run_cypher(node_query)
    edges_raw, _ = _run_cypher(edge_query)

    # Build node id set for safety
    node_ids = {n["id"] for n in nodes_raw}
    edges = [
        {"source": e["source"], "target": e["target"],
         "rel_type": e["rel_type"], "weight": e["weight"]}
        for e in edges_raw
        if e["source"] in node_ids and e["target"] in node_ids
    ]

    return {"nodes": nodes_raw, "edges": edges}


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND INGESTION TASK
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(session_id: str, topic: str, max_papers: int):
    """Runs in a background thread."""
    sess = sessions[session_id]
    sess["status"] = "ingesting"
    sess["log"]    = []
    sess["papers"] = []
    sess["ingested_papers"] = []
    sess["failed_papers"] = []
    sess["error"] = None
    _ensure_observations(sess)
    ingest_started_at = time.perf_counter()

    def log(msg: str):
        sess["log"].append(msg)

    try:
        _append_ingestion_event(sess, "ingestion_started", status="running", details={
            "topic": topic,
            "max_papers": max_papers,
        })

        # 1. Clear DB
        log("🗑️  Clearing database...")
        clear_started = time.perf_counter()
        clear_database()
        _append_ingestion_event(sess, "clear_database", latency_ms=_elapsed_ms(clear_started))
        log("✅ Database cleared")

        # 2. Setup schema
        log("🔧 Setting up schema...")
        schema_started = time.perf_counter()
        setup_schema()
        _append_ingestion_event(sess, "setup_schema", latency_ms=_elapsed_ms(schema_started))
        log("✅ Schema ready")

        # 3. Fetch papers
        log(f"🔍 Searching ArXiv for: {topic}")
        fetch_started = time.perf_counter()
        arxiv_client = arxiv.Client()
        search = arxiv.Search(query=topic, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for result in arxiv_client.results(search):
            aid = re.sub(r"v\d+$", "", result.entry_id.split("/abs/")[-1])
            papers.append({
                "arxiv_id":  aid,
                "title":     result.title,
                "abstract":  result.summary.replace("\n", " "),
                "authors":   [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "year":      result.published.year,
                "pdf_url":   result.pdf_url,
                "category":  result.categories[0] if result.categories else "",
            })
            log(f"📄 Found: {result.title[:70]}")

        _append_ingestion_event(sess, "fetch_arxiv_results", latency_ms=_elapsed_ms(fetch_started), details={
            "paper_count": len(papers),
        })
        log(f"✅ Fetched {len(papers)} paper(s)")
        sess["papers"] = papers

        # 4. Process each paper
        for i, paper in enumerate(papers, 1):
            paper_started = time.perf_counter()
            log(f"\n📖 [{i}/{len(papers)}] Processing: {paper['title'][:60]}...")

            try:
                log("   🤖 Extracting entities with Gemini...")
                extract_started = time.perf_counter()
                extraction = extract_entities(paper)
                _append_ingestion_event(sess, "extract_entities", latency_ms=_elapsed_ms(extract_started), details={
                    "paper_id": paper["arxiv_id"],
                    "concepts": len(extraction.concepts_introduced) + len(extraction.concepts_applied),
                    "methods": len(extraction.methods),
                    "datasets": len(extraction.datasets),
                    "topics": len(extraction.topics),
                    "sample_methods": [m.name for m in extraction.methods[:3]],
                })

                log(f"   ✅ Concepts: {len(extraction.concepts_introduced)+len(extraction.concepts_applied)} | "
                    f"Methods: {len(extraction.methods)} | Topics: {len(extraction.topics)}")

                log("   📥 Downloading PDF...")
                pdf_started = time.perf_counter()
                try:
                    resp = requests.get(paper["pdf_url"], timeout=30, headers={"User-Agent": "PaperGraph/2.0"})
                    resp.raise_for_status()
                    sections, cited_refs = parse_sections(resp.content)
                    _append_ingestion_event(sess, "parse_pdf_sections", latency_ms=_elapsed_ms(pdf_started), details={
                        "paper_id": paper["arxiv_id"],
                        "sections": len(sections),
                        "citations": len(cited_refs),
                    })
                    log(f"   ✅ Sections: {len(sections)} | Citations: {len(cited_refs)}")
                except Exception as e:
                    log(f"   ⚠️  PDF failed ({e}), continuing without sections")
                    _append_ingestion_event(sess, "parse_pdf_sections", status="error", latency_ms=_elapsed_ms(pdf_started), details={
                        "paper_id": paper["arxiv_id"],
                        "error": _truncate_text(str(e), 280),
                    })
                    sections, cited_refs = [], []

                log("   💾 Writing to Neo4j...")
                write_started = time.perf_counter()
                ingest_paper(paper, extraction, sections, cited_refs)
                _append_ingestion_event(sess, "write_to_neo4j", latency_ms=_elapsed_ms(write_started), details={
                    "paper_id": paper["arxiv_id"],
                    "sections_written": len(sections),
                })

                sess["ingested_papers"].append({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "year": paper["year"],
                })
                log(f"   ✅ Done: {paper['arxiv_id']}")

                _append_ingestion_event(sess, "paper_completed", latency_ms=_elapsed_ms(paper_started), details={
                    "paper_id": paper["arxiv_id"],
                    "title": _truncate_text(paper["title"], 120),
                })

            except Exception as paper_error:
                err_msg = _truncate_text(str(paper_error), 320)
                sess["failed_papers"].append({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "error": err_msg,
                })
                _append_ingestion_event(sess, "paper_failed", status="error", latency_ms=_elapsed_ms(paper_started), details={
                    "paper_id": paper["arxiv_id"],
                    "title": _truncate_text(paper["title"], 120),
                    "error": err_msg,
                })
                log(f"   ❌ Failed: {paper['arxiv_id']} ({err_msg})")
                continue

            time.sleep(2)

        ingested_count = len(sess["ingested_papers"])
        failed_count = len(sess["failed_papers"])

        if ingested_count == 0:
            sess["status"] = "error"
            sess["error"] = "No papers were ingested successfully. Check logs/observation for details."
            _append_ingestion_event(sess, "ingestion_failed", status="error", latency_ms=_elapsed_ms(ingest_started_at), details={
                "papers_fetched": len(papers),
                "papers_ingested": ingested_count,
                "papers_failed": failed_count,
            })
            log(f"❌ {sess['error']}")
            return

        # 5. Build agent (if at least one paper ingested)
        log("\n🤖 Building RAG agent...")
        agent_started = time.perf_counter()
        chat_ready = False
        try:
            sess["agent"] = build_agent()
            sess["thread_id"] = f"session_{session_id}"
            sess["messages"] = []
            chat_ready = True
            _append_ingestion_event(sess, "build_rag_agent", latency_ms=_elapsed_ms(agent_started))
        except Exception as build_error:
            build_err_msg = _truncate_text(str(build_error), 320)
            sess["agent"] = None
            sess["thread_id"] = None
            sess["messages"] = []
            _append_ingestion_event(sess, "build_rag_agent", status="error", latency_ms=_elapsed_ms(agent_started), details={
                "error": build_err_msg,
            })
            log(f"⚠️ Agent build failed ({build_err_msg}). Graph will still be available.")

        if failed_count > 0 or not chat_ready:
            sess["status"] = "ready_with_errors"
            issue_bits = [f"ingested {ingested_count}/{len(papers)} papers"]
            if failed_count > 0:
                issue_bits.append(f"{failed_count} failed")
            if not chat_ready:
                issue_bits.append("chat unavailable")
            sess["error"] = "Partial ingestion: " + ", ".join(issue_bits) + "."
            _append_ingestion_event(sess, "ingestion_completed_with_errors", status="error", latency_ms=_elapsed_ms(ingest_started_at), details={
                "papers_fetched": len(papers),
                "papers_ingested": ingested_count,
                "papers_failed": failed_count,
                "chat_ready": chat_ready,
            })
            log(f"⚠️ {sess['error']}")
        else:
            sess["status"] = "ready"
            sess["error"] = None
            _append_ingestion_event(sess, "ingestion_completed", latency_ms=_elapsed_ms(ingest_started_at), details={
                "papers_fetched": len(papers),
                "papers_ingested": ingested_count,
            })
            log("🎉 Ingestion complete! You can now ask questions.")

    except Exception as e:
        sess["status"] = "error"
        sess["error"]  = str(e)
        _append_ingestion_event(sess, "ingestion_failed", status="error", latency_ms=_elapsed_ms(ingest_started_at), details={
            "error": _truncate_text(str(e), 420),
        })
        log(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


class IngestRequest(BaseModel):
    topic: str
    max_papers: int = 3


@app.post("/api/session/new")
async def new_session(req: IngestRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "status":    "pending",
        "topic":     req.topic,
        "log":       [],
        "papers":    [],
        "ingested_papers": [],
        "failed_papers": [],
        "agent":     None,
        "thread_id": None,
        "messages":  [],
        "error":     None,
    }
    _ensure_observations(sessions[session_id])
    background_tasks.add_task(run_ingestion, session_id, req.topic, min(req.max_papers, MAX_PAPERS))
    return {"session_id": session_id}


@app.get("/api/session/{session_id}/status")
async def session_status(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return {
        "status":  sess["status"],
        "log":     sess["log"],
        "papers":  [{"arxiv_id": p["arxiv_id"], "title": p["title"], "year": p["year"]}
                    for p in sess.get("papers", [])],
        "ingested_papers": sess.get("ingested_papers", []),
        "failed_papers": sess.get("failed_papers", []),
        "chat_ready": bool(sess.get("agent")),
        "error":   sess.get("error"),
    }


@app.get("/api/session/{session_id}/graph")
async def session_graph(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess.get("status") in {"pending", "ingesting"}:
        raise HTTPException(400, "Ingestion not complete")
    if len(sess.get("ingested_papers", [])) == 0:
        raise HTTPException(400, "No successfully ingested papers available")
    return get_graph_data()


class ChatRequest(BaseModel):
    query: str


@app.post("/api/session/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest):
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess.get("status") in {"pending", "ingesting"}:
        raise HTTPException(400, "Ingestion not complete")
    if not sess.get("agent"):
        raise HTTPException(400, "Chat is unavailable for this session (agent was not built).")

    agent     = sess["agent"]
    thread_id = sess["thread_id"]
    chat_started = time.perf_counter()
    turn_id = _start_chat_turn(sess, req.query)

    config = {"configurable": {"thread_id": thread_id}}
    obs_token = OBS_CONTEXT.set({"session_id": session_id, "turn_id": turn_id})
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                {"query": req.query, "messages": []},
                config=config,
            ),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        answer = result.get("answer") if isinstance(result, dict) else None
        if not answer:
            raise ValueError("Agent returned an empty answer.")
        _complete_chat_turn(
            session_id,
            turn_id,
            status="ok",
            answer=answer,
            latency_ms=_elapsed_ms(chat_started),
        )
    except asyncio.TimeoutError:
        timeout_msg = (
            f"Chat timed out after {CHAT_TIMEOUT_SECONDS}s. "
            "Try a shorter or more specific query."
        )
        _complete_chat_turn(
            session_id,
            turn_id,
            status="error",
            error=timeout_msg,
            latency_ms=_elapsed_ms(chat_started),
        )
        raise HTTPException(504, timeout_msg)
    except Exception as e:
        _complete_chat_turn(
            session_id,
            turn_id,
            status="error",
            error=str(e),
            latency_ms=_elapsed_ms(chat_started),
        )
        raise HTTPException(500, f"Chat failed: {e}")
    finally:
        OBS_CONTEXT.reset(obs_token)

    sess["messages"].append({"role": "user",      "content": req.query})
    sess["messages"].append({"role": "assistant", "content": answer})

    return {"answer": answer}


@app.get("/api/session/{session_id}/messages")
async def get_messages(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return {"messages": sess.get("messages", [])}


@app.get("/api/session/{session_id}/observations")
async def get_observations(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    obs = _ensure_observations(sess)
    return {
        "status": sess.get("status"),
        "meta": obs.get("meta", {}),
        "totals": obs.get("totals", {}),
        "ingestion_events": obs.get("ingestion_events", [])[-300:],
        "chat_turns": obs.get("chat_turns", [])[-50:],
    }


# startup
@app.on_event("startup")
async def startup():
    setup_schema()