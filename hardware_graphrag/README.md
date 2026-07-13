# Hardware Specification GraphRAG

A Streamlit application that converts hardware specification documents
(PDF / DOCX / Markdown) into a searchable **Knowledge Graph** (Neo4j) and
**Vector Database** (FAISS), then answers natural-language questions using
**hybrid retrieval** (graph traversal + semantic search) grounded in the
source document.

```
Upload → Parse → Hierarchical Chunk → [Vector DB]
                                    → Entity Extraction → Relationship Extraction
                                    → Entity Normalization → [Neo4j Knowledge Graph]
                                                                     │
                                                     Hybrid Retrieval ◄──┘
                                                                     │
                                                          LLM Answer Generation
```

## Features

- **Document Parsing** — chapters, sections, subsections, tables, figures,
  timing diagrams, register descriptions, notes, warnings, examples; every
  section preserves page number, chapter, section number, parent/prev/next.
- **Hierarchical Chunking** — never fixed-size; splits by chapter → section →
  subsection, only falling back to sentence-boundary splitting for very long
  subsections, with 10–20% overlap between adjacent chunks.
- **Entity Extraction** — Protocol, Interface, Channel, Signal, Register,
  Field, Transaction, TimingConstraint, ClockDomain, MemoryRegion, Address,
  Interrupt, State, Command, DataStructure, Feature.
- **Relationship Extraction** — BELONGS_TO, PART_OF, USES, DEPENDS_ON,
  HANDSHAKES_WITH, CONNECTS_TO, ASSERTED_BEFORE, ASSERTED_AFTER,
  TRANSITIONS_TO, GENERATES, REQUIRES, CONFIGURES, READS_FROM, WRITES_TO,
  REFERENCES, DEFINED_IN.
- **Entity Normalization** — merges duplicate names (e.g. "Write Address",
  "Write Address Channel", "WA Channel" → `WRITE_ADDRESS_CHANNEL`).
- **Hybrid Retrieval** — detects entities in the question, expands the
  knowledge graph neighborhood, merges with semantic vector search results,
  and grounds the LLM's answer with citations.
- **Pluggable LLM** — OpenAI, Groq, or Gemini; works with **no LLM configured
  at all** by falling back to rule-based extraction and extractive answers.
- **Pluggable Graph backend** — real Neo4j, or an in-memory graph store if
  Neo4j isn't available, so the app is usable with zero external services.

## Project Structure

```
app.py                          # Streamlit entry point (run this)
config.py                       # Central configuration (env-var driven)
core/
  parsing/                      # PDF / DOCX / Markdown parsers + heading heuristics
  chunking/                     # Hierarchical chunker
  extraction/                   # LLM client + entity/relationship extractors
  normalization/                # Entity de-duplication & canonicalization
  graph/                        # Neo4j builder + in-memory fallback graph store
  vectorstore/                  # FAISS-backed semantic search + embedding cache
  retrieval/                    # Hybrid (graph + vector) retriever
  pipeline.py                   # Orchestrates all steps end-to-end
  registry.py                   # On-disk cache of processed docs/chunks/entities
ui/                              # One module per Streamlit page
utils/
  models.py                     # Shared dataclasses (Chunk, Entity, Relationship, ...)
  logger.py                     # Centralized logging (console + file + in-app Logs page)
data/                            # Runtime storage: uploads, vector index, graph cache, logs
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env with your API keys / Neo4j creds
streamlit run app.py
```

The app works immediately with **no configuration** — it will use:
- Heuristic regex-based entity/relationship extraction instead of an LLM.
- An in-memory graph instead of Neo4j.
- A deterministic hashing embedder instead of `sentence-transformers`
  (if that package isn't installed).

To unlock full LLM-based extraction and answer generation, set in `.env`:

```
LLM_PROVIDER=openai        # or groq / gemini
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
```

To connect a real Neo4j instance:

```
NEO4J_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

> Note: some Neo4j Cypher queries in `core/graph/neo4j_builder.py` use APOC
> procedures (`apoc.path.subgraphAll`, `apoc.coll.toSet`). Install the APOC
> plugin on your Neo4j instance, or simplify those queries to core Cypher
> if APOC isn't available.

## Pages

| Page | Purpose |
|---|---|
| Dashboard | Document/chunk/entity/relationship counts, graph & vector stats |
| Document Upload | Upload PDF/DOCX/Markdown, run the pipeline with live progress |
| Chunk Explorer | Browse/filter/search hierarchical chunks |
| Entity Explorer | Browse/filter extracted entities, view merged aliases |
| Knowledge Graph | Interactive graph visualization (streamlit-agraph), node inspector |
| Vector Search | Semantic search over indexed chunks with similarity scores |
| Question Answering | Ask questions; see retrieved graph nodes, chunks, and grounded answer |
| Logs | Parsing / extraction / graph / vector / error logs |

## Notes on scalability & incremental updates

- Both the Neo4j and in-memory graph stores use **MERGE-based upserts**, so
  re-processing a document (or adding a new one) incrementally updates the
  graph rather than duplicating nodes.
- Embeddings are cached to disk (keyed by a hash of the chunk text), so
  re-running the pipeline on unchanged content skips re-embedding.
- Large documents are processed chunk-by-chunk with progress callbacks, so
  the Streamlit UI stays responsive and shows granular progress.
