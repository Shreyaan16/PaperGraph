# PaperGraph 🔬

A FastAPI application that ingests ArXiv research papers into a Neo4j knowledge graph and enables multi-turn RAG conversations over the data.

## Features

- **Topic-based ingestion** — Enter any research topic, fetch 2–3 ArXiv papers
- **Session isolation** — Each new session clears the DB and starts fresh
- **Interactive knowledge graph** — Force-directed D3.js visualization with all node types (Paper, Author, Concept, Method, Dataset, Topic) and their relationships
- **Multi-turn RAG chat** — LangGraph agent with Cypher + vector retrieval, conversation history per session
- **Live ingestion log** — Real-time progress streamed to the sidebar
- **Observation tab** — Per-node latency, token usage, estimated cost, query/response trace, and ingestion step outputs

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_database_name
HF_TOKEN=your_huggingface_token
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=papergraph
MODEL_INPUT_COST_PER_1M_USD=0.3
MODEL_OUTPUT_COST_PER_1M_USD=2.5
```

`MODEL_INPUT_COST_PER_1M_USD` and `MODEL_OUTPUT_COST_PER_1M_USD` are optional and used for estimated token cost display in the Observation tab.

### 2. Install Dependencies (uv + pyproject.toml)

```bash
uv sync
```

### 3. Run

```bash
uv run uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Run via Docker Hub

Use this when you want to run the app directly from your published image (without local Python setup).

### 1. Keep your `.env` file in the project folder

Use the same variables shown in the **Environment Variables** section above.

### 2. Pull your image from Docker Hub

```bash
docker pull shreyaan16/papergraph:latest
```

### 3. Run the container

```bash
docker run --name papergraph --rm -p 8000:8000 --env-file .env shreyaan16/papergraph:latest
```

Then open:
- http://localhost:8000 (UI)
- http://localhost:8000/docs (API docs)

### 4. Optional: Run in detached mode

```bash
docker run -d --name papergraph -p 8000:8000 --env-file .env shreyaan16/papergraph:latest
docker logs -f papergraph
```

### 5. Stop / remove

```bash
docker stop papergraph
```

If you used `--rm`, the container is removed automatically after stop.

### 6. Update to latest image

```bash
docker pull shreyaan16/papergraph:latest
docker stop papergraph
docker run --name papergraph --rm -p 8000:8000 --env-file .env shreyaan16/papergraph:latest
```

## Usage

1. **Enter a topic** in the sidebar (e.g. "attention mechanisms in transformers")
2. **Select paper count** (2 or 3)
3. **Click "Ingest Papers"** — watch the live log
4. **View the Knowledge Graph** tab to explore the extracted entities
   - Click any node to see details
   - Drag nodes to rearrange
   - Scroll to zoom
5. **Switch to Research Chat** and ask questions:
   - "Who are the authors?"
   - "What methods are used?"
   - "Summarize the key contributions"
   - "What datasets are evaluated on?"
6. **Open the Observation tab** to inspect:
  - ingestion step-by-step outputs and timings
  - LangGraph node latency and outputs
  - per-turn query/response trace
  - token usage and estimated cost

## Architecture

```
FastAPI
├── POST /api/session/new          → start ingestion (background task)
├── GET  /api/session/{id}/status  → poll ingestion progress
├── GET  /api/session/{id}/graph   → fetch graph data for visualization
├── POST /api/session/{id}/chat    → multi-turn RAG query
├── GET  /api/session/{id}/messages→ conversation history
└── GET  /api/session/{id}/observations → ingestion + node-level telemetry

Ingestion Pipeline:
  ArXiv API → Gemini entity extraction → PDF parse → Neo4j write

RAG Agent (LangGraph):
  query_analyzer → cypher_generator → cypher_checker (retry loop)
  → graph_retriever → result_checker → vector_retriever
  → context_builder → answer_generator
```

## Neo4j Graph Schema

### Nodes
| Label   | Key Properties |
|---------|---------------|
| Paper   | arxiv_id, title, abstract, summary, year, category |
| Author  | name, name_norm |
| Section | section_id, title, content, embedding (768-dim) |
| Concept | name, name_norm, description |
| Method  | name, name_norm, description |
| Dataset | name, name_norm, description |
| Topic   | name, name_norm, description |

### Relationships
`AUTHORED_BY`, `CO_AUTHORED`, `HAS_SECTION`, `NEXT_SECTION`, `INTRODUCES`, `APPLIES`, `USES_METHOD`, `USES_DATASET`, `BELONGS_TO`, `RELATED_TO`, `EXTENDS`, `CONTRADICTS`, `CITES`
