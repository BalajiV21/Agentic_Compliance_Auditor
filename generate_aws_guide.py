"""
Generate AWS deployment guide PDF for the Agentic Compliance Auditor project.
Run once to produce AWS_Deployment_Guide.pdf in the project root.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    ListFlowable, ListItem, KeepTogether, Preformatted,
)


OUTPUT = "AWS_Deployment_Guide.pdf"

# ----- Styles -----
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=14,
                    textColor=HexColor("#0B3D91"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=15, spaceBefore=14,
                    spaceAfter=8, textColor=HexColor("#0B3D91"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12, spaceBefore=10,
                    spaceAfter=6, textColor=HexColor("#333333"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5,
                      leading=14, alignment=TA_JUSTIFY, spaceAfter=8)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9.5, leading=13,
                      textColor=HexColor("#444444"), leftIndent=12, rightIndent=12,
                      backColor=HexColor("#F4F6FA"), borderPadding=6)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, leading=11,
                      textColor=HexColor("#111111"), backColor=HexColor("#F2F2F2"),
                      leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=8,
                      borderPadding=6)
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=26,
                       textColor=HexColor("#0B3D91"))
SUBTITLE = ParagraphStyle("SubTitle", parent=styles["Normal"], fontSize=13,
                          textColor=HexColor("#555555"), alignment=TA_CENTER)


def code(text: str):
    return Preformatted(text.strip("\n"), CODE)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(i, BODY), leftIndent=10) for i in items],
        bulletType="bullet", start="circle", leftIndent=16,
    )


def kv_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths or [1.8 * inch, 4.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F7F9FC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ----- Build content -----
story = []

# Title page
story += [
    Spacer(1, 1.5 * inch),
    Paragraph("AWS Deployment Guide", TITLE),
    Spacer(1, 0.2 * inch),
    Paragraph("Agentic Compliance Auditor", SUBTITLE),
    Spacer(1, 0.15 * inch),
    Paragraph("Architecture &middot; Code changes &middot; Step-by-step setup", SUBTITLE),
    Spacer(1, 2.5 * inch),
    Paragraph("Prepared for: self-hosted deployment of a RAG pipeline using "
              "FastAPI, ChromaDB, Redis, and Ollama on Amazon Web Services.",
              BODY),
    PageBreak(),
]

# Table of contents (manual)
story += [
    Paragraph("Contents", H1),
    bullets([
        "1. Architecture overview",
        "2. Deployment options (EC2, ECS, EKS)",
        "3. Pre-deployment code changes",
        "4. Step-by-step EC2 deployment",
        "5. Containerized deployment (Docker + ECS)",
        "6. Switching Ollama to Amazon Bedrock",
        "7. Managed services: ElastiCache, OpenSearch, S3",
        "8. Security hardening",
        "9. Monitoring &amp; observability",
        "10. Cost estimates",
        "11. Troubleshooting checklist",
    ]),
    PageBreak(),
]

# ---- 1. Architecture ----
story += [
    Paragraph("1. Architecture overview", H1),
    Paragraph(
        "The application consists of four logical tiers:",
        BODY,
    ),
    bullets([
        "<b>UI tier</b> &mdash; Streamlit (<i>ui/streamlit_app.py</i>), calls the API over HTTP.",
        "<b>API tier</b> &mdash; FastAPI (<i>src/api/main.py</i>), handles queries, memory, ingestion.",
        "<b>Retrieval / agent tier</b> &mdash; ChromaDB + LangGraph agent running in-process "
        "with the API. Uses Ollama for LLM inference by default.",
        "<b>State tier</b> &mdash; Redis (session memory) and the ChromaDB persistent directory.",
    ]),
    Paragraph(
        "For AWS, each tier maps to a managed or compute service. The simplest path is a "
        "single EC2 instance running everything; the scalable path separates the LLM, "
        "vector store, and API into dedicated services.",
        BODY,
    ),
    Paragraph("Tier-to-service mapping", H3),
    kv_table([
        ["Tier", "Recommended AWS service"],
        ["UI (Streamlit)", "EC2 / App Runner / Amplify (behind CloudFront)"],
        ["API (FastAPI)", "EC2 / ECS Fargate / App Runner"],
        ["LLM inference", "Amazon Bedrock (recommended) or Ollama on EC2 g5"],
        ["Vector store", "ChromaDB on EBS, or Amazon OpenSearch w/ k-NN"],
        ["Session memory", "Amazon ElastiCache for Redis"],
        ["Document storage", "Amazon S3 (raw docs), EBS (processed DB)"],
        ["Secrets", "AWS Secrets Manager / SSM Parameter Store"],
        ["Logs &amp; metrics", "CloudWatch Logs, CloudWatch Metrics"],
    ]),
    PageBreak(),
]

# ---- 2. Deployment options ----
story += [
    Paragraph("2. Deployment options", H1),
    Paragraph(
        "Three viable paths, in order of complexity. Pick based on the audience and load.",
        BODY,
    ),

    Paragraph("Option A &mdash; Single EC2 instance (recommended for demo/PoC)", H2),
    bullets([
        "One EC2 instance runs FastAPI, Streamlit, Ollama, Redis, and ChromaDB (files on EBS).",
        "<b>Pros:</b> Simplest setup, lowest cost, one box to debug.",
        "<b>Cons:</b> No HA, LLM on CPU is slow; upgrade to g5.xlarge for GPU acceleration.",
        "<b>Instance:</b> <i>g5.xlarge</i> (1x A10G, 16 GB GPU) for real-time inference, "
        "or <i>c6i.2xlarge</i> for CPU-only.",
    ]),

    Paragraph("Option B &mdash; ECS Fargate for API + managed AI services", H2),
    bullets([
        "API and UI run as Fargate tasks behind an Application Load Balancer.",
        "LLM is served by Amazon Bedrock (no self-hosted Ollama).",
        "Vector store moves to OpenSearch Serverless (k-NN).",
        "Redis via ElastiCache, documents in S3.",
        "<b>Pros:</b> Fully managed, auto-scaling, no GPU to manage.",
        "<b>Cons:</b> More services to configure, higher baseline cost.",
    ]),

    Paragraph("Option C &mdash; EKS (Kubernetes)", H2),
    bullets([
        "Only if you already run Kubernetes. Overkill for this project.",
        "Same components as Option B but packaged as pods and deployments.",
    ]),

    Paragraph(
        "The rest of this guide details Option A end-to-end, then shows how to migrate "
        "individual pieces to managed services (Option B).",
        NOTE,
    ),
    PageBreak(),
]

# ---- 3. Pre-deployment code changes ----
story += [
    Paragraph("3. Pre-deployment code changes", H1),
    Paragraph(
        "The codebase already works on localhost. These changes make it production-ready "
        "before you ship the image or bake the AMI.",
        BODY,
    ),

    Paragraph("3.1 Use absolute paths anchored to the project root", H3),
    Paragraph(
        "Already applied in config/config.py: CHROMA_PERSIST_DIR, LOG_FILE, and "
        "EVAL_DATASET_PATH resolve relative to the config file so the app works from any "
        "working directory. Keep this.",
        BODY,
    ),

    Paragraph("3.2 Replace @app.on_event with lifespan", H3),
    Paragraph(
        "The FastAPI deprecation warning becomes a hard error in newer versions. Update "
        "<i>src/api/main.py</i>:",
        BODY,
    ),
    code("""
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    global vector_store, retriever, agent, memory
    vector_store = VectorStore(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name="compliance_documents",
        embedding_model=settings.EMBEDDING_MODEL,
    )
    retriever = CitationRetriever(vector_store, top_k=settings.TOP_K_RESULTS)
    tools = create_langchain_tools(retriever)
    agent = ComplianceAgent(retriever, tools,
                            model_name=settings.OLLAMA_MODEL,
                            max_iterations=settings.MAX_ITERATIONS,
                            enable_reflection=settings.ENABLE_SELF_REFLECTION)
    try:
        memory = RedisMemory(host=settings.REDIS_HOST,
                             port=settings.REDIS_PORT,
                             db=settings.REDIS_DB)
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}); using in-memory buffer")
        memory = ConversationBufferMemory()
    yield
    # --- shutdown ---
    logger.info("API shutting down cleanly")

app = FastAPI(title="Agentic Compliance Auditor API",
              version="1.0.0", lifespan=lifespan)
"""),

    Paragraph("3.3 Lock down CORS", H3),
    code("""
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "https://your-ui-domain.example.com")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
"""),

    Paragraph("3.4 Remove reload=True and bind workers from settings", H3),
    code("""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,     # 2-4 per vCPU
        log_level=settings.LOG_LEVEL.lower(),
    )
"""),

    Paragraph("3.5 Secure the /ingest endpoint", H3),
    Paragraph(
        "The current endpoint accepts an arbitrary file_path query parameter and reads from "
        "disk. Before deployment, either remove the endpoint or require an authenticated "
        "S3 URI:",
        BODY,
    ),
    code("""
@app.post("/ingest")
async def ingest_document(s3_uri: str, bg: BackgroundTasks,
                          token: str = Depends(verify_api_token)):
    assert s3_uri.startswith("s3://"), "Only s3:// URIs allowed"
    bg.add_task(ingest_from_s3, s3_uri)
    return {"status": "queued", "s3_uri": s3_uri}
"""),

    Paragraph("3.6 Add API-key authentication", H3),
    code("""
from fastapi import Depends, Header, HTTPException
import os, secrets

API_TOKEN = os.environ["API_TOKEN"]

def verify_api_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    if not secrets.compare_digest(authorization.split()[1], API_TOKEN):
        raise HTTPException(403, "Invalid token")
    return True

# Add Depends(verify_api_token) to every POST/DELETE route.
"""),

    Paragraph("3.7 Structured logging to stdout", H3),
    Paragraph(
        "CloudWatch Logs captures stdout/stderr from EC2 (via agent) and Fargate "
        "(automatically). Configure loguru:",
        BODY,
    ),
    code("""
from loguru import logger
import sys, json

logger.remove()
logger.add(sys.stdout, serialize=True, level=os.environ.get("LOG_LEVEL", "INFO"))
"""),

    Paragraph("3.8 Health and readiness endpoints", H3),
    code("""
@app.get("/healthz")  # liveness — always 200 if process alive
async def healthz(): return {"ok": True}

@app.get("/readyz")   # readiness — checks dependencies
async def readyz():
    try:
        assert vector_store.get_collection_stats()["total_chunks"] > 0
        if isinstance(memory, RedisMemory):
            memory.client.ping()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, str(e))
"""),

    PageBreak(),
]

# ---- 4. Step-by-step EC2 deployment ----
story += [
    Paragraph("4. Step-by-step EC2 deployment (Option A)", H1),

    Paragraph("4.1 Provision the instance", H3),
    bullets([
        "Region: pick the closest to your users (e.g. <i>us-east-1</i>).",
        "AMI: Ubuntu 22.04 LTS (ami-0xxx).",
        "Instance type: <b>g5.xlarge</b> for GPU Ollama, or <b>c6i.2xlarge</b> for CPU-only "
        "(plus Bedrock for LLM).",
        "Storage: 100 GB gp3 EBS. Increase if documents &gt; 5 GB.",
        "Security group: inbound 22 (SSH from your IP), 8000 (API, from ALB SG only), "
        "8501 (UI, from your IP / VPN / CloudFront only).",
        "IAM role: attach a role with S3 read, Bedrock:InvokeModel, Secrets Manager read, "
        "and CloudWatch Logs write.",
    ]),

    Paragraph("4.2 Base setup", H3),
    code("""
# SSH in
ssh ubuntu@<PUBLIC_IP>

sudo apt update && sudo apt -y upgrade
sudo apt install -y python3.11 python3.11-venv python3-pip git redis-server nginx
sudo systemctl enable --now redis-server

# Verify
redis-cli ping   # -> PONG
"""),

    Paragraph("4.3 Install Ollama (skip if using Bedrock)", H3),
    code("""
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &            # starts the daemon
ollama pull llama3.2      # ~2 GB
"""),

    Paragraph("4.4 Clone and configure the app", H3),
    code("""
cd /opt
sudo git clone <YOUR_REPO_URL> compliance-auditor
sudo chown -R ubuntu:ubuntu compliance-auditor
cd compliance-auditor

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
"""),

    Paragraph("4.5 Create the production .env", H3),
    Paragraph(
        "Fetch secrets from Secrets Manager rather than hard-coding. Example <i>.env</i>:",
        BODY,
    ),
    code("""
# LLM
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2

# Vector DB
EMBEDDING_MODEL=all-mpnet-base-v2

# Redis (local)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Agent
MAX_ITERATIONS=3
ENABLE_SELF_REFLECTION=false

# Auth &amp; logging
LOG_LEVEL=INFO
API_TOKEN=<generate-with-openssl-rand-hex-32>
ALLOWED_ORIGIN=https://your-ui-domain.example.com
"""),

    Paragraph("4.6 Ingest documents", H3),
    code("""
source venv/bin/activate
# Pull raw docs from S3
aws s3 sync s3://your-bucket/regulations ./data/sample_docs

python -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.')
from config import settings
from retrieval.vector_store import VectorStore
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import RegulationChunker
vs=VectorStore(settings.CHROMA_PERSIST_DIR,'compliance_documents',
               embedding_model=settings.EMBEDDING_MODEL)
loader=DocumentLoader(); chunker=RegulationChunker(512,0)
docs=loader.load_directory('./data/sample_docs')
chunks=[c for d in docs for c in chunker.chunk_document(d)]
vs.add_documents(chunks); print(vs.get_collection_stats())
"
"""),

    Paragraph("4.7 Run FastAPI with Gunicorn + systemd", H3),
    code("""
# /etc/systemd/system/compliance-api.service
[Unit]
Description=Compliance Auditor API
After=network.target redis-server.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/compliance-auditor
EnvironmentFile=/opt/compliance-auditor/.env
ExecStart=/opt/compliance-auditor/venv/bin/gunicorn \\
    -k uvicorn.workers.UvicornWorker \\
    -w 4 -b 0.0.0.0:8000 \\
    --chdir /opt/compliance-auditor/src/api main:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""),
    code("""
sudo systemctl daemon-reload
sudo systemctl enable --now compliance-api
sudo systemctl status compliance-api
curl http://127.0.0.1:8000/healthz
"""),

    Paragraph("4.8 Run Streamlit UI as a service", H3),
    code("""
# /etc/systemd/system/compliance-ui.service
[Unit]
Description=Compliance Auditor UI
After=network.target compliance-api.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/compliance-auditor
EnvironmentFile=/opt/compliance-auditor/.env
ExecStart=/opt/compliance-auditor/venv/bin/streamlit run ui/streamlit_app.py \\
    --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
"""),

    Paragraph("4.9 TLS + reverse proxy with nginx + Let's Encrypt", H3),
    code("""
sudo apt install -y certbot python3-certbot-nginx

# /etc/nginx/sites-available/auditor
server {
  listen 80;
  server_name auditor.example.com;
  location / { proxy_pass http://127.0.0.1:8501; }         # UI
  location /api/ { proxy_pass http://127.0.0.1:8000/; }    # API
}

sudo ln -s /etc/nginx/sites-available/auditor /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d auditor.example.com
"""),

    PageBreak(),
]

# ---- 5. Docker / ECS ----
story += [
    Paragraph("5. Containerized deployment (Docker + ECS)", H1),

    Paragraph("5.1 Dockerfile for the API", H3),
    code("""
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \\
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV PYTHONPATH=/app/src:/app
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4",
     "-b", "0.0.0.0:8000", "--chdir", "src/api", "main:app"]
"""),

    Paragraph("5.2 Build &amp; push to ECR", H3),
    code("""
aws ecr create-repository --repository-name compliance-auditor-api
aws ecr get-login-password | docker login --username AWS \\
   --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
docker build -t compliance-auditor-api .
docker tag compliance-auditor-api:latest \\
   <acct>.dkr.ecr.us-east-1.amazonaws.com/compliance-auditor-api:latest
docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/compliance-auditor-api:latest
"""),

    Paragraph("5.3 ECS task definition (key fields)", H3),
    bullets([
        "<b>launchType</b>: FARGATE",
        "<b>cpu/memory</b>: 2 vCPU / 8 GB (API-only, with Bedrock for LLM)",
        "<b>environment</b>: REDIS_HOST, API_TOKEN, ALLOWED_ORIGIN, etc. "
        "(pull secrets from Secrets Manager via <i>secrets</i> block)",
        "<b>logConfiguration</b>: awslogs driver &rarr; /ecs/compliance-auditor",
        "<b>portMappings</b>: 8000/tcp",
    ]),

    Paragraph("5.4 Front with an ALB", H3),
    bullets([
        "Target group: HTTP 8000, health-check path <i>/healthz</i>.",
        "Listener: HTTPS 443 with an ACM certificate. Redirect HTTP 80 &rarr; 443.",
        "Security group: only ALB SG can reach the ECS service SG.",
    ]),

    Paragraph("5.5 ChromaDB in a container", H3),
    Paragraph(
        "ChromaDB's file-based persistence needs durable storage. Options:",
        BODY,
    ),
    bullets([
        "<b>EFS mount</b> on the ECS task &mdash; simplest, supports multiple replicas.",
        "<b>Run ChromaDB in server mode</b> on a separate EC2 / ECS task, using its HTTP API.",
        "<b>Migrate to OpenSearch Serverless</b> with k-NN index &mdash; fully managed (see &sect;7).",
    ]),

    PageBreak(),
]

# ---- 6. Bedrock ----
story += [
    Paragraph("6. Switching Ollama to Amazon Bedrock", H1),
    Paragraph(
        "Bedrock removes the need to run Ollama yourself, eliminates GPU costs, and gives "
        "you access to Claude, Llama, Mistral, and Titan models via a single API. Code "
        "changes in <i>src/agents/compliance_agent.py</i>:",
        BODY,
    ),
    code("""
# Before:
from langchain_ollama import ChatOllama
self.llm = ChatOllama(model=model_name, temperature=0.1)

# After (Bedrock):
from langchain_aws import ChatBedrock
self.llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    model_kwargs={"temperature": 0.1, "max_tokens": 1024},
)
"""),
    bullets([
        "Install: <i>pip install langchain-aws boto3</i>",
        "IAM: instance role needs <i>bedrock:InvokeModel</i> for the chosen model ARN.",
        "Enable the model in Bedrock &rarr; Model Access in the console (one-time).",
        "Update your <i>.env</i>: remove OLLAMA_BASE_URL / OLLAMA_MODEL, add "
        "<i>AWS_REGION=us-east-1</i> and <i>BEDROCK_MODEL_ID=&lt;arn&gt;</i>.",
    ]),
    PageBreak(),
]

# ---- 7. Managed services ----
story += [
    Paragraph("7. Managed services: ElastiCache, OpenSearch, S3", H1),

    Paragraph("7.1 Amazon ElastiCache (Redis)", H3),
    bullets([
        "Create a Redis cluster (cache.t4g.small is enough for sessions).",
        "Attach the cluster's security group to allow port 6379 from the API SG.",
        "Update <i>.env</i>: REDIS_HOST=&lt;primary-endpoint&gt;, REDIS_PORT=6379.",
        "Enable AUTH + TLS &mdash; then set REDIS_TLS=true and pass "
        "<i>ssl=True, password=&lt;AUTH_TOKEN&gt;</i> in <i>RedisMemory.__init__</i>.",
    ]),

    Paragraph("7.2 Amazon OpenSearch Serverless (vector store)", H3),
    Paragraph(
        "Replace ChromaDB with managed k-NN search &mdash; recommended once you have &gt; "
        "100k vectors or need HA.",
        BODY,
    ),
    bullets([
        "Create an OpenSearch Serverless collection (type: <i>Vector search</i>).",
        "Create an index with an <i>hnsw</i> field matching your embedding dim (768 for "
        "mpnet).",
        "Write a thin adapter in <i>src/retrieval/opensearch_store.py</i> implementing "
        "the same interface as VectorStore (<i>add_documents</i>, <i>search</i>, "
        "<i>hybrid_search</i>, <i>get_collection_stats</i>). The rest of the app remains "
        "unchanged.",
    ]),
    code("""
# Sketch:
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   os.environ["AWS_REGION"], "aoss",
                   session_token=credentials.token)
client = OpenSearch(hosts=[{"host": ENDPOINT, "port": 443}], http_auth=awsauth,
                    use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection)
"""),

    Paragraph("7.3 S3 for document ingestion", H3),
    bullets([
        "Upload raw regulation PDFs / TXTs to s3://your-bucket/regulations/",
        "The /ingest endpoint downloads via boto3 and passes to DocumentLoader.",
        "Enable S3 bucket versioning and Object Lock if you need an audit trail.",
    ]),

    PageBreak(),
]

# ---- 8. Security ----
story += [
    Paragraph("8. Security hardening", H1),
    bullets([
        "<b>Secrets:</b> never commit .env. Store API_TOKEN, REDIS_AUTH, Bedrock model ARNs "
        "in Secrets Manager. Inject at container start via <i>secrets</i> block (ECS) or "
        "<i>aws secretsmanager</i> on EC2 boot.",
        "<b>IAM:</b> separate roles for ingestion (S3 read only) and runtime (Bedrock invoke, "
        "Redis SG, CloudWatch write). No *:* policies.",
        "<b>Network:</b> put API + data tier in private subnets; only the ALB sits in "
        "public subnets.",
        "<b>TLS everywhere:</b> ACM cert on ALB, internal traffic over VPC (still fine in "
        "HTTP within private subnets but prefer HTTPS).",
        "<b>Request limits:</b> in nginx / ALB, set body size 1 MB, rate-limit /query to "
        "30 req/min per IP.",
        "<b>Input validation:</b> cap query length (e.g. 4 KB) in the Pydantic model.",
        "<b>Dependency pinning:</b> pin requirements.txt exact versions; scan with "
        "<i>pip-audit</i> or Amazon Inspector.",
        "<b>Patch cadence:</b> rebuild the image weekly; use EC2 Image Builder for AMI "
        "refresh.",
    ]),
    PageBreak(),
]

# ---- 9. Monitoring ----
story += [
    Paragraph("9. Monitoring &amp; observability", H1),
    bullets([
        "<b>CloudWatch Logs:</b> agent streams stdout. Create a metric filter on "
        "&quot;ERROR&quot; &rarr; alarm &rarr; SNS &rarr; email/Slack.",
        "<b>CloudWatch Metrics:</b> expose latency + retrieval count via a Prometheus "
        "<i>prometheus-fastapi-instrumentator</i>, scrape with a sidecar or forward to "
        "CloudWatch.",
        "<b>ALB access logs</b> &rarr; S3, queryable via Athena.",
        "<b>Synthetics:</b> CloudWatch Synthetic canary hitting /readyz every minute.",
        "<b>Cost monitoring:</b> AWS Budgets alarm at 80% of expected monthly.",
    ]),
    PageBreak(),
]

# ---- 10. Cost estimates ----
story += [
    Paragraph("10. Cost estimates (us-east-1, on-demand, USD/month)", H1),
    Paragraph("Rough ballparks; always confirm with the AWS Pricing Calculator.", BODY),
    kv_table([
        ["Option / Component", "Estimated monthly cost"],
        ["Option A &mdash; g5.xlarge (GPU Ollama) 24/7",  "~$730 (compute) + $10 EBS + $0 Redis local = ~$740"],
        ["Option A &mdash; c6i.2xlarge (CPU-only, no LLM)",  "~$250 + $10 EBS = ~$260"],
        ["Option A + Bedrock (Claude Sonnet)",  "~$260 EC2 + usage-based ($3/M input tokens)"],
        ["Option B &mdash; Fargate 2 tasks (0.5 vCPU / 1 GB)",  "~$35"],
        ["Option B &mdash; ALB + data transfer",  "~$20"],
        ["Option B &mdash; OpenSearch Serverless (2 OCUs min)",  "~$350"],
        ["Option B &mdash; ElastiCache t4g.small",  "~$22"],
        ["Option B &mdash; Bedrock Claude Sonnet, 1M input tok/day",  "~$90"],
        ["S3 storage + egress (5 GB + 50 GB out)",  "~$5"],
        ["CloudWatch Logs (1 GB)",  "~$1"],
    ], col_widths=[3.2 * inch, 3.0 * inch]),
    Paragraph(
        "<b>Punch line:</b> if you only need inference occasionally, Option B with Bedrock "
        "is cheapest in absolute terms and scales to zero Fargate load. If the LLM is hit "
        "constantly, a reserved g5.xlarge or Savings Plan often beats Bedrock per token.",
        NOTE,
    ),
    PageBreak(),
]

# ---- 11. Troubleshooting ----
story += [
    Paragraph("11. Troubleshooting checklist", H1),
    kv_table([
        ["Symptom", "Likely cause &amp; fix"],
        ["API returns 502 at the ALB", "Target group unhealthy &mdash; check /healthz; ensure "
         "SG on tasks allows ALB SG; confirm the container listens on 0.0.0.0 not 127.0.0.1."],
        ["&quot;Found 0 results&quot; in logs", "Ingestion ran with a different EMBEDDING_MODEL "
         "than the API. Re-ingest after wiping chroma_db."],
        ["Scores &gt; 100%", "Keyword / rerank boost applied without clamp. See "
         "vector_store.py hybrid_search and retriever.py _rerank_results &mdash; both must "
         "use min(1.0, ...) after boosting."],
        ["Redis connection refused", "Security group on ElastiCache doesn't allow the API SG; "
         "or REDIS_HOST still pointing at 127.0.0.1."],
        ["Ollama timeout on EC2", "GPU drivers missing &mdash; use the Deep Learning AMI or "
         "install NVIDIA CUDA drivers. Confirm with <i>nvidia-smi</i>."],
        ["Chunker hangs", "Small sections trigger the overlap infinite loop. Apply the "
         "chunker fix (forward-progress guard) or run with chunk_overlap=0."],
        ["CORS error in browser", "ALLOWED_ORIGIN in .env does not match the UI URL exactly "
         "(no trailing slash, include the scheme)."],
        ["High Bedrock bill", "Reduce top_k, trim retrieved_docs before sending to LLM, "
         "cache identical queries in Redis for 5 min."],
    ], col_widths=[2.0 * inch, 4.2 * inch]),
]

# ----- Build -----
doc = SimpleDocTemplate(
    OUTPUT, pagesize=letter,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    title="AWS Deployment Guide — Agentic Compliance Auditor",
    author="Compliance Auditor Project",
)
doc.build(story)
print(f"Wrote {OUTPUT}")
