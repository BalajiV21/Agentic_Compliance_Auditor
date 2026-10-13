# Agentic Compliance Auditor — Interview Q&A

Read top to bottom. Each section: **main questions → short answers → follow-ups**.
Answers match the actual code in this repo, so they hold up if the interviewer opens GitHub.

---

## 0. Numbers to remember

| Item | Value |
|---|---|
| Documents | 3 (GDPR, HIPAA, SOC2) |
| Chunks | ~246 (resume) |
| Chunk size / overlap | **512 characters / 50 characters (~10%)** — characters, not tokens |
| Embedding model | OpenAI `text-embedding-3-small`, 1536 dimensions |
| LLM | `gpt-4o-mini`, temperature 0.1 |
| Vector DB | ChromaDB, cosine distance, HNSW index, stored on disk |
| Top-k sent to LLM | 5 chunks |
| Graph nodes | retrieve → generate → (tools) → reflect → loop or END |
| RAGAS scores (10 questions) | Faithfulness **0.88**, Answer relevancy **0.90**, Context precision **0.81**, Context recall **0.66** |
| Server | AWS EC2 t3.medium (2 vCPU, 4 GB), Amazon Linux 2023, Elastic IP |
| Serving | nginx :80 → uvicorn/FastAPI :8000, managed by systemd |
| Session memory | Redis, 1-hour TTL, in-memory fallback (last 10 messages) |
| OpenAI timeout / retries | 30 s on embeddings; client retries twice with backoff by default |

---

## 1. Project overview

**Q: Explain your project in 30 seconds.**
A: Compliance teams spend hours checking regulations like GDPR, HIPAA and SOC2 by hand, and a plain LLM makes up legal details without citing sources. I built an agentic RAG system. It retrieves the relevant regulation text with hybrid search, generates an answer grounded only in that text with article-level citations, then reviews its own answer and loops back if it's weak. It's built with LangGraph, ChromaDB and OpenAI, served by FastAPI with streaming, has a React frontend with a PDF viewer that highlights citations, runs on AWS EC2, and is evaluated with RAGAS.

**Q: What problem does it solve? Why not just use ChatGPT?**
A: ChatGPT answers from memory. It can hallucinate, it doesn't know a company's private policy documents, and it doesn't cite sources. In compliance every claim must be traceable, so answers must come from the actual documents, with citations a person can click and verify.

**Q: What makes it "agentic" and not just RAG?**
A: Plain RAG is a straight line: retrieve, then generate, then stop. My system **makes decisions**. The LLM can call tools for more searches, and a reflection step judges the answer and can send it back to retrieval. LangGraph's conditional edges make that loop possible, and `max_iterations` caps it.

**Q: Walk me through the architecture.**
A: Offline: documents → loader → structure-aware chunker → OpenAI embeddings → ChromaDB. Online: React → nginx → FastAPI → LangGraph agent (retrieve → generate → reflect) → answer with citations, streamed back through SSE, with session history saved in Redis.

*Follow-ups*
- **What was the hardest part?** Getting retrieval to land on the exact article. Semantic search returned Article 18 for an Article 17 question, which is why I added a keyword layer. Also a chunker infinite-loop bug (section 2).
- **What would you do differently?** A true BM25 index fused with RRF, structured reflection output, a LangGraph checkpointer, and Docker from day one.

---

## 2. Ingestion and chunking

**Q: How do documents get into the system?**
A: `setup.py` runs `DocumentLoader` (PDF, TXT, DOCX). It tags each document's type (GDPR/HIPAA/SOC2) from the filename, chunks it with `RegulationChunker`, embeds the chunks with OpenAI in batches, and stores vectors plus metadata in ChromaDB. It's a one-time step. New documents can be added through `/ingest`.

**Q: Why chunk at all?**
A: An embedding turns a whole text into one vector. Embed a whole document and the vector averages every topic, so it matches nothing well. Smaller chunks give precise matches, fit the LLM's context, cost less, and let a citation point to one article.

**Q: Which chunking strategy did you use, and why?**
A: Structure-aware recursive chunking. First, regex finds legal headers (`Article 17 -`, `Section 164.502 -`, `CC6.1 -`) and each section is chunked separately, so a chunk never mixes two articles. Inside long sections it cuts near 512 characters at the most natural break: section divider → paragraph → newline → `; ` (legal list items) → sentence → space. Regulations already have structure, so I use it instead of guessing.

**Q: Why not fixed-size or semantic chunking?**
A: Fixed-size cuts through sentences and mixes articles. Semantic chunking (splitting where embeddings change) costs extra embedding calls, and it's unnecessary when the documents already have clear article boundaries.

**Q: What metadata does each chunk carry?**
A: `document_type`, `section`, `page_number`, `start_char`/`end_char`, `chunk_index`, and `legal_references` (for example `["Article 17"]`). These are used for metadata filtering, citations and PDF highlighting.

**Q: Why 512 characters?**
A: One regulation clause is usually 200–600 characters, so one chunk is roughly one clause. Five chunks come to about 600 tokens of context, which is cheap and fast. The section metadata adds back the context a small chunk loses.

**Q: What if chunks are too small? Too big?**
A: Too small: precise, but loses context ("(b) consent withdrawn;" doesn't say it's about erasure). You need more chunks per answer, and **recall drops**. Too big: the embedding blurs many topics, the LLM gets noise and more tokens, citations get vague, and **precision drops**.

**Q: What overlap did you use, and why?**
A: 50 characters, about 10%, so a sentence cut at a boundary still appears whole in one chunk. It's on the low end of the usual 10–20% because the splitter already cuts at natural boundaries. More overlap adds duplicate text, cost, and near-duplicate results in the top 5.

**Q: What happens with 0% or 50% overlap?**
A: 0%: facts on a boundary get split and become hard to retrieve. 50%: almost everything is stored twice, the index grows, and the top-k fills with near-duplicates, so the LLM sees less variety.

*Follow-ups*
- **A bug you fixed?** An infinite loop in the chunker. When paragraph breaks were very close together, stepping back by the overlap landed on the same split point forever. The fix forces every step forward by at least `chunk_size / 4` (`MIN_STEP`).
- **Characters vs tokens?** Characters today. For production I'd size by tokens with `tiktoken`, since LLM limits and costs are in tokens.
- **Improvements?** Parent-child chunking (search small chunks, send the whole article), and contextual headers ("GDPR > Article 17:") prepended before embedding.
- **A known limitation?** For PDFs, chunking is page by page (to keep page numbers) and skips article splitting, so an article that crosses a page break gets split. The sample corpus is TXT, which does use section splitting.

---

## 3. Embeddings and vector database

**Q: What is an embedding?**
A: A list of numbers representing the meaning of text. Similar meanings give vectors that point in similar directions. I use `text-embedding-3-small`, which gives 1536 dimensions and is cheap (about $0.02 per 1M tokens).

**Q: What is cosine similarity?**
A: The angle between two vectors. 1 means the same direction (same meaning), 0 means unrelated. ChromaDB is configured with `hnsw:space: cosine`, and I convert distance to similarity as `1 − distance`.

**Q: What is HNSW?**
A: Hierarchical Navigable Small World, a layered graph index for fast **approximate** nearest-neighbour search, so a query doesn't compare against every vector.

**Q: Why ChromaDB?**
A: It's embedded, needs no separate server, saves to disk, stores metadata, and supports `where` filters. That's right for a few hundred chunks. At millions of vectors, or across multiple servers, I'd move to OpenSearch k-NN, pgvector, or Pinecone.

**Q: How does metadata filtering work?**
A: `VectorStore.search()` turns a filter like `{"document_type": "GDPR"}` into a ChromaDB `where` clause (`$eq` or `$in`), so only matching chunks are searched. It's used by the tools and the `/search` endpoint.

*Follow-ups*
- **Why OpenAI embeddings and not a local model?** Better quality, no GPU or model to host, low cost. A local sentence-transformers model is an option if data mustn't leave the network.
- **What if you change embedding models?** Everything must be re-embedded. Vectors from different models aren't comparable.
- **Batching?** Embeddings are sent in batches of 100 during setup, about 60× fewer network round-trips.

---

## 4. Hybrid search, BM25 and RRF

**Q: What is hybrid search?**
A: Combining semantic search (meaning) with keyword search (exact words). Users ask vaguely ("can a customer make me delete their data?") or exactly ("GDPR Article 17"). Semantic search handles the first, keyword search the second.

**Q: Why isn't semantic search enough?**
A: Embeddings blur exact identifiers. "Article 17" and "Article 18" are both about data-subject rights and sit close together in vector space, so the wrong article can rank first.

**Q: Why isn't keyword search enough?**
A: It's blind to synonyms. "Delete" won't match "erasure".

**Q: What is BM25?**
A: Best Matching 25, the standard keyword-ranking formula (the Elasticsearch/Lucene default). It scores every document using: (1) **term frequency** with diminishing returns (parameter k1), (2) **IDF**, so rare words like "erasure" count more than common words like "data", and (3) **length normalization** (parameter b), so long documents don't win just by being long.

**Q: BM25 vs TF-IDF?**
A: BM25 adds TF saturation and length normalization, which is why it replaced TF-IDF as the standard.

**Q: Does BM25 filter results?**
A: No. It **scores every chunk**, then we sort and keep the top N (for example 20). The N is our choice.

**Q: Does it match exact words?**
A: Yes, exact tokens after lowercasing. "erase" won't match "erasure" without stemming, which is why it's paired with semantic search.

**Q: What does hybrid do in YOUR code? (be honest)**
A: It's BM25-*style*. Semantic search fetches the top candidates from ChromaDB, I extract keywords from the query (stop words and words of 2 letters or fewer removed), and boost each candidate's score by 3% per matching keyword, then re-sort and keep the top 5. It fixed exact-article lookups. The limit is that it only re-ranks what semantic search already found, with no IDF and no separate index. That's part of why recall is 0.66.

**Q: How would you build it properly?**
A: Run semantic search and a real BM25 index (`rank_bm25`) **separately**, each over all chunks, each returning its own top 20, then merge them with **Reciprocal Rank Fusion**: score = Σ 1/(60 + rank). Optionally a cross-encoder reranker on the top 20.

**Q: Why RRF and not just adding the scores?**
A: Cosine similarity (0–1) and BM25 scores (0–20+) are on different scales, and ranks are comparable. A chunk ranked well by **both** methods rises to the top. 60 is the standard constant that stops the #1 position from dominating.

**Q: What's a reranker?**
A: A cross-encoder reads the query and a chunk **together** and scores their relevance. It's more accurate than comparing vectors but slower, so it's used only on the top 20. My code has a heuristic placeholder (`_rerank_results`) that is off by default.

*Follow-ups*
- **Small code gotcha:** in `hybrid` mode the metadata filter isn't passed through, so filtering works only in `semantic` and `keyword` modes. Numbers like "17" are also dropped by the length rule in keyword extraction.
- **The `keyword` strategy:** scores `0.6 × semantic + 0.4 × keyword count`. It exists, but the agent uses `hybrid`.

---

## 5. LangGraph agent and self-reflection

**Q: What is LangGraph?**
A: A library for building agents as a **state machine**: nodes are steps, edges are transitions, conditional edges are decisions, and a shared state object flows through.

**Q: Why LangGraph instead of a LangChain chain?**
A: A chain is a straight line. Self-correction needs a **cycle** (reflect → retrieve) plus explicit stop conditions. LangGraph supports loops, conditional routing, tool nodes, streaming each step, and checkpointing.

**Q: What's in the state?**
A: `query`, `retrieved_docs`, `messages`, `answer`, `reflection`, `needs_reflection`, `iterations`, `max_iterations` (a TypedDict).

**Q: Walk through the nodes.**
A:
1. **retrieve**: hybrid retrieval, top 5 chunks.
2. **generate**: gpt-4o-mini with tools bound. The system prompt says: answer only from context, cite the article or section, say if it isn't in the documents.
3. **tools** (if the LLM requests one): `search_regulations`, `search_specific_article`, `cross_reference_regulations`, `extract_requirements`. Then back to generate.
4. **reflect**: a second LLM call checks whether claims are cited, whether anything is speculative, contradictory or missing, and whether it answers the question.

**Q: What are the conditional edges?**
A: After generate: tool call → `tools`. Otherwise `reflect` if enabled, else END. After reflect: needs improvement and `iterations < max_iterations` → back to `retrieve`. Otherwise END.

**Q: How do you prevent infinite loops?**
A: `max_iterations`. Each reflection increments `iterations`, and once it reaches the max the graph ends regardless.

**Q: How does reflection decide "needs improvement"? (be honest)**
A: Today it looks for words like "missing", "incomplete", "should" or "incorrect" in the critique text. It's simple, but it can fire on positive reviews too. I'd switch to **structured output**, JSON like `{grounded: bool, missing_citations: [], revised_query: ""}`, and route on the boolean.

**Q: Does reflection check against the sources?**
A: Currently the reflection prompt receives the **answer only**, not the retrieved chunks. The improvement is to pass the chunks so it verifies each claim against the text, and to rewrite the query before retrying (right now a retry reuses the same query).

**Q: Why temperature 0.1?**
A: Compliance answers should be factual and repeatable, not creative.

**Q: How do you reduce hallucination?**
A: Grounding-only prompt, an instruction to say "not in the documents", low temperature, hybrid retrieval for the right chunks, the reflection check, clickable citations for human verification, and RAGAS faithfulness to measure it.

*Follow-ups*
- **Cost of reflection?** At least 2 extra LLM calls per loop. It's controlled by `max_iterations` and the `ENABLE_SELF_REFLECTION` flag.
- **Reflection retry vs failure retry?** Reflection is a *quality* loop that's part of the agent logic. Failure retry is the OpenAI client re-sending a call that errored.
- **Tool calling?** The LLM returns `tool_calls`. LangGraph's `ToolNode` executes them and appends the results as messages, then generate runs again.

---

## 6. FastAPI

**Q: What is FastAPI, and why use it?**
A: A Python web framework for APIs. It's async (fast), validates requests automatically with Pydantic, generates Swagger docs at `/docs`, and uses type hints.

**Q: What endpoints do you have?**
A: `POST /query` (sync), `POST /query/stream` (SSE), `GET /search`, `GET /regulations`, `GET /stats`, `GET /health`, `GET/DELETE /conversation/{id}`, `POST /ingest` (background task).

**Q: How does request validation work?**
A: A Pydantic `BaseModel` (`QueryRequest`: `query: str`, `session_id: Optional[str]`, `document_type: Optional[str]`). Missing or wrong fields get an automatic 422.

**Q: Write a basic endpoint.**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    return {"answer": agent.run(req.query)}
```
Run it with `uvicorn main:app --reload --port 8000`.

**Q: Path vs query parameters?**
A: `/regulations/{name}` puts `name` in the path. `?limit=5` is a query parameter. Both are typed, converted and validated.

**Q: Status codes?**
A: 200 OK, 400 bad request, 401/403 auth, 404 not found, 422 validation error, 429 rate limited, 500 server error.

**Q: How do you load heavy objects?**
A: Once at startup (`@app.on_event("startup")`): the vector store, retriever, agent and memory. The modern way is a `lifespan` context manager.

**Q: async def vs def?**
A: `async def` lets the server handle other requests while one is waiting on I/O, but only if the code inside awaits async calls. My endpoint is `async def` but calls the synchronous `agent.run()`, which blocks the event loop. The fix is `def` (FastAPI runs it in a thread pool) or `await graph.ainvoke()`.

**Q: What is uvicorn?**
A: The ASGI server that actually listens on the port and runs the FastAPI app. In production, Gunicorn with Uvicorn workers gives multiple processes.

---

## 7. Streaming (SSE)

**Q: What does `/query/stream` do?**
A: It keeps one HTTP connection open and pushes events as each LangGraph node finishes: `start → retrieved → generating → reflecting → answer → done` (or `error`). The UI shows a live trace.

**Q: Is it token-by-token?**
A: No. It streams **steps**, and the final answer arrives in one `answer` event. That's deliberate: reflection can reject a draft, so streaming draft tokens would show text that might get thrown away.

**Q: How is it implemented?**
A: An async generator `yield`s dicts like `{"event": ..., "data": json}` inside `EventSourceResponse` (sse-starlette), looping over `agent.stream()`, which yields after each node. `return` sends once and ends. `yield` sends a piece and keeps going.

**Q: How would you add token streaming?**
A: Call OpenAI with `stream=True` (or LangChain's `astream`) in the final answer step and `yield` each token as an event.

**Q: SSE vs WebSocket?**
A: SSE is server → client only, plain HTTP, simple, and passes through nginx easily. WebSocket is two-way, which is needed only for interactive two-way apps.

**Q: Why `fetchEventSource` on the frontend?**
A: The browser's built-in `EventSource` supports only GET. My endpoint needs POST with a JSON body.

**Q: Any nginx gotcha for SSE?**
A: `proxy_buffering off`. Otherwise nginx holds the events and sends them all at the end.

---

## 8. nginx

**Q: Why nginx?**
A: Two jobs: (1) serve the built React static files fast, and (2) reverse-proxy `/api/*` to FastAPI on `127.0.0.1:8000`.

**Q: Benefits of a reverse proxy?**
A: Same origin, so no CORS problems. Port 8000 stays private. Fast static file serving. One place to add HTTPS, rate limiting and load balancing later.

**Q: Show the config idea.**
```nginx
location /     { root /var/www/dist; try_files $uri /index.html; }
location /api/ { proxy_pass http://127.0.0.1:8000/; proxy_buffering off; }
```

---

## 9. Deployment (AWS)

**Q: How is it deployed?**
A: A single EC2 t3.medium on Amazon Linux 2023 with an Elastic IP. nginx on port 80 serves React and proxies `/api` to uvicorn/FastAPI on localhost:8000, which runs as a **systemd** service (starts at boot, restarts on crash). ChromaDB is files on disk, Redis runs locally, and the LLM and embeddings come from OpenAI, so no GPU is needed.

**Q: Deployment steps?**
A: Launch EC2 → Elastic IP → Security Group (80 open, 22 only from my IP, 8000 and 6379 closed) → install Python, Node, nginx, Redis → git clone, venv, pip install → `.env` with the API key (never in Git) → `python setup.py` → `npm run build` → nginx config → systemd service → test.

**Q: What is systemd?**
A: Linux's service manager. `Restart=always` restarts on crash, `enable` starts at boot, and `journalctl -u <service>` shows the logs.

**Q: Why t3.medium? Why no GPU?**
A: The model runs at OpenAI. The server only routes requests, searches ChromaDB and talks to Redis. 2 vCPU and 4 GB is enough and cheap.

**Q: Why EC2 and not Lambda?**
A: Agent runs take 10+ seconds and stream over SSE, ChromaDB needs persistent disk, and Lambda has timeouts and cold starts.

**Q: How do you deploy updates?**
A: Manually today: SSH, `git pull`, reinstall if needed, rebuild the frontend, `systemctl restart`. Next step: GitHub Actions CI/CD and Docker.

**Q: Is it dockerized?**
A: Not this project yet. Docker Compose (API + Redis + nginx + vector DB) is the planned next step, which also makes moving to ECS Fargate easy. My fraud-detection project is dockerized.

**Q: Where is the API key stored?**
A: In `.env` on the server, git-ignored. In production: AWS Secrets Manager, read through an IAM role attached to the instance, with no keys on disk.

*Honesty note:* `docs/AWS_Deployment_Guide.pdf` describes an older version (Ollama, Streamlit, Ubuntu, GPU). Describe the live deployment above.

---

## 10. Scaling

**Q: What happens when traffic increases? What breaks first?**
A: Not the CPU. About 95% of each request is spent waiting on OpenAI. What breaks first: (1) blocking sync code in one worker, so requests queue; (2) OpenAI rate limits (429); (3) cost; (4) a single server with no high availability; (5) local ChromaDB can't be shared across servers.

**Q: How would you scale it?**
A:
- **Stage 1, same server:** Gunicorn with several Uvicorn workers, a truly async agent (`ainvoke` plus the async OpenAI client), lower `max_iterations`.
- **Stage 2, reduce load:** Redis response cache for repeated questions, per-user rate limiting, retry with backoff, timeouts.
- **Stage 3, horizontal:** stateless API containers (ECS Fargate or an Auto Scaling Group) behind an Application Load Balancer, ElastiCache for sessions, a shared vector DB (OpenSearch, pgvector, or Chroma server mode), S3 for documents, React on S3 + CloudFront, HTTPS through ACM, CloudWatch alarms.
- **Stage 4:** an ingestion queue (SQS), model routing (cheap model for easy questions), a higher OpenAI tier or Bedrock.

**Q: What does "stateless" mean, and why does it matter?**
A: The server keeps nothing important locally. Sessions live in Redis and vectors in a shared DB, so any request can go to any server and you can add or remove servers freely.

**Q: Why Redis for sessions?**
A: Fast key-value lookup per `session_id`, with automatic expiry (1-hour TTL), and it can be shared across servers. The app falls back to in-memory storage if Redis is down.

**Q: What latency does a query have?**
A: About 4–15 seconds: embedding (~0.2 s), Chroma (~0.05 s), generate (2–5 s), reflect (1–3 s), plus any loops.

---

## 11. Failure handling

**Q: If a query fails halfway, does it retry or resume?**
A: It doesn't resume. Each OpenAI call retries automatically (twice by default, with exponential backoff, for network errors, 429 and 5xx), and embeddings have a 30-second timeout. If a step still fails, the exception goes up. `/query` returns a 500, and `/query/stream` sends an `error` event. The UI shows which steps finished, and the user resubmits, which **starts again from retrieve**, because the graph state lives only in memory (the graph is compiled without a checkpointer).

**Q: How would you make it resume?**
A: Compile the graph with a **LangGraph checkpointer** (SQLite/Postgres/Redis saver) and run with `config={"configurable": {"thread_id": session_id}}`. State is saved after each node. On failure, `graph.invoke(None, config)` continues from the last successful node, with no repeated LLM calls.

**Q: Other resilience patterns?**
A: Per-node `RetryPolicy`. Graceful degradation (return the unverified draft if only reflection fails). Model fallback (`llm.with_fallbacks([...])`). Timeouts on all calls. A circuit breaker. A retry button in the UI.

**Q: Is retrying safe?**
A: Yes. Search and LLM calls are read-only (idempotent), so a retry only costs money. The one write is saving to Redis. Note that the user message is stored before the run, so a failed run leaves an unanswered message in the history. I'd store it after success or mark it as failed.

---

## 12. Evaluation (RAGAS)

**Q: How do you evaluate the system?**
A: RAGAS on 10 hand-written questions with reference answers (6 GDPR, 2 HIPAA, 2 SOC2). The script runs the agent, collects question, answer, retrieved contexts and reference, and RAGAS scores 4 metrics with an LLM judge. Results go to `docs/ragas_report.json` and `.csv`. About $0.05 per run.

**Q: Who is the judge?**
A: A normal general-purpose **gpt-4o-mini** through the regular OpenAI API, at temperature 0. It isn't specially trained. RAGAS supplies the judging prompts and scoring math, and breaks each metric into small yes/no checks that a general LLM handles reliably.

**Q: Explain each metric.**
A:
- **Faithfulness (0.88):** split the answer into claims and check each against the retrieved chunks. Supported ÷ total. This is the hallucination check.
- **Answer relevancy (0.90):** the judge writes questions the answer would fit, which are embedded and compared with the real question. This is the on-topic check.
- **Context precision (0.81):** are the retrieved chunks relevant, and ranked near the top? Ranking-weighted.
- **Context recall (0.66):** split the reference into facts and check how many appear in the retrieved chunks. Did we retrieve everything needed?

**Q: Which metrics need a reference answer?**
A: Context precision and context recall. Faithfulness and relevancy don't, so they can run on live traffic.

**Q: What did the results tell you?**
A: Generation is strong and retrieval recall is the bottleneck, especially for **list questions**. "SOC 2 Trust Service Criteria" scored precision 0 and recall 0 (a retrieval failure). "HIPAA administrative safeguards" had recall 0. "GDPR Article 5 principles" had recall 0.5, because the list is spread across several chunks. Fixes: parent-document retrieval, higher top-k for list questions, true BM25 + RRF.

**Q: Why did Article 17 get faithfulness 0.7 when the answer was correct?**
A: The model added true GDPR facts that weren't in the retrieved chunks. RAGAS counts those as unsupported. For compliance that's the right behaviour: answers should come only from the documents.

**Q: Limitations of LLM-as-judge?**
A: The judge can be wrong, and it's the same model family as the generator, so it may be lenient. 10 questions is small (one question moves the average about 10%). Scores vary a little between runs. Mitigations: a stronger judge (gpt-4o) or a fine-tuned hallucination model (Patronus Lynx, Vectara HHEM), human spot-checks, a larger test set including "not in the documents" questions, and averaging several runs.

**Q: How would you use evaluation in production?**
A: As a CI regression gate (fail the build if faithfulness drops below a threshold), for parameter sweeps (chunk size, overlap, top-k, strategy), with a deterministic citation-accuracy metric (cited article numbers must appear in the retrieved chunks' `legal_references`), and with sampled live-traffic scoring plus user feedback.

*Honesty note:* the evaluation script runs with `enable_reflection=False`, so these scores measure RAG **without** the reflection loop. The repo has no "reflection on vs off" comparison and no citation-accuracy metric. Be ready to explain exactly how the resume's **56% → 82% citation accuracy** was measured, or quote the RAGAS numbers instead.

---

## 13. Frontend

**Q: What does the UI do?**
A: React + Vite + TypeScript + Tailwind, with a Zustand store. A split screen shows the regulation PDF on the left (react-pdf/pdf.js) and the live agent trace and answer on the right. Citation chips like `[1] p.2` jump to the page and highlight the cited passage. Tabs switch between GDPR, HIPAA and SOC2.

**Q: Why highlight citations?**
A: Traceability. A compliance officer can verify each claim in one click, which builds trust.

**Q: How do you develop the UI without the backend?**
A: `VITE_USE_MOCK=true` swaps in a mock event stream (`mockStream.ts`), while `realStream.ts` talks to the backend.

**Q: Why Zustand?**
A: A small global store shared across components (query, trace, chunks, answer, error) without Redux boilerplate.

*Note:* `ui/streamlit_app.py` was the earlier prototype UI. That's why Streamlit appears on the resume.

---

## 14. Security

**Q: What security measures exist, and what would you add?**
A: Today: the API key in a git-ignored `.env`, backend ports closed in the Security Group, SSH restricted to my IP. For production:
- **HTTPS** (Let's Encrypt or an ACM certificate). The demo currently runs on HTTP.
- **Secrets Manager + IAM role** (least privilege) instead of `.env`.
- **CORS**: currently `allow_origins=["*"]`. Restrict it to the real domain.
- **Authentication**: API key or JWT, especially on write endpoints.
- **`/ingest` accepts a file path from the request**, which could read any server file. Remove it or allow only authenticated S3 uploads.
- Rate limiting, input length limits, and SSM Session Manager instead of SSH.
- **Prompt injection**: a malicious document could contain instructions. Mitigate by treating retrieved text as data in the prompt, allowing only approved tools, and validating outputs.

**Q: Data privacy?**
A: Documents, embeddings and sessions stay on the server. Only the composed prompt (question plus retrieved chunks) goes to OpenAI, and embedding calls send the chunk text. For sensitive data: OpenAI's zero-retention options, Azure OpenAI or Bedrock inside a private network, or local models.

---

## 15. Cost

**Q: How much does a query cost?**
A: gpt-4o-mini is about $0.15 per 1M input tokens. A query uses roughly 1–3k tokens per LLM call and 2–6 calls, so a fraction of a cent. Embeddings are almost free. The EC2 instance is about $30/month. A full RAGAS run is about $0.05.

**Q: How would you reduce cost?**
A: Cache repeated questions, skip reflection for simple questions, lower `max_iterations`, send fewer and smaller chunks, route easy questions to a cheaper model.

---

## 16. Limitations and improvements (have this list ready)

1. True BM25 index + RRF fusion + a cross-encoder reranker → better recall.
2. Structured reflection output, reflection that sees the source chunks, and query rewriting on retry.
3. LangGraph checkpointer for resume, plus per-node retry policies.
4. A truly async agent and multiple workers.
5. Parent-document retrieval for list questions.
6. Token-based chunk sizing, with page mapping for PDFs.
7. Docker Compose and CI/CD, HTTPS, auth, locked-down CORS, a secured `/ingest`.
8. A larger evaluation set, a reflection on/off comparison, a citation-accuracy metric, and evaluation in CI.
9. Unit tests (the chunker loop bug is the kind a pytest suite would catch).

---

## 17. Behavioural questions tied to the project

**Q: Tell me about a bug you fixed.**
A: The chunker went into an infinite loop on documents with closely spaced paragraph breaks. After cutting at a separator, it stepped back by the overlap and found the same separator again. I traced it with logging, then enforced a minimum forward step (`chunk_size / 4`) and accepted a split only if it moves forward meaningfully. Lesson: loops with overlap need a guaranteed progress condition, and a unit test should cover it.

**Q: A decision you'd change?**
A: I'd build the evaluation harness first. Measuring retrieval early would have pointed me to the recall problem and to true hybrid search sooner.

**Q: How did you decide what to build?**
A: Started with plain RAG, measured it, saw exact-article lookups fail, added keyword boosting, then added reflection to catch uncited claims, and used RAGAS to see where the remaining weakness is (recall).

---

## 18. Resume claims: check before the interview

| Resume line | What the repo shows | What to do |
|---|---|---|
| "Raw regulatory PDFs" | The indexed corpus is `.txt` samples. The PDFs in `frontend/public` are generated for the viewer. The loader supports PDF. | Say "regulatory documents (PDF/TXT/DOCX supported)". |
| "246 chunks across 3 documents" | Not verifiable without running | Run `python setup.py` or call `/stats` and confirm the number. |
| "Self-verification loop… re-checks against retrieved sources" | Reflection sees only the answer, not the sources | Fix the code, or say "reviews its own answer for grounding and citations". |
| "Citation accuracy 56% → 82%" | No citation metric. Evaluation runs with reflection off. | Know how you measured it, or quote the RAGAS scores. |
| "Hybrid… semantic, keyword, metadata filters" | True, but keyword matching is a boost on semantic results, not a separate BM25 index | Say "BM25-style keyword scoring", and mention RRF as the next step. |
| "RAGAS-based evaluation framework" | True: 4 metrics, 10 questions | Fine. Mention the citation metric as a next step. |
| README says sentence-transformers embeddings | The code uses OpenAI `text-embedding-3-small` | Fix the README line. |

---

## 19. 60-second final pitch

> "I built an agentic RAG system for regulatory compliance across GDPR, HIPAA and SOC2. Documents are split with a structure-aware chunker on article boundaries: 512 characters, 10% overlap, rich metadata. They're embedded with OpenAI and stored in ChromaDB. At query time, a LangGraph agent retrieves with hybrid semantic and keyword search, generates a cited answer with gpt-4o-mini restricted to the retrieved text, and runs a reflection step that can loop back if the answer is weak, capped by max iterations. FastAPI serves it with Server-Sent Events so the React UI shows each step live and highlights cited passages in the PDF. It runs on EC2 behind nginx with systemd. RAGAS gives faithfulness 0.88 and relevancy 0.90, and context recall of 0.66 tells me retrieval is the next thing to improve, through true BM25 with rank fusion and parent-document retrieval."
