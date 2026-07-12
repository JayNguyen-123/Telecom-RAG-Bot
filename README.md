# Telecom-RAG-Bot
Enterprise Telecom RAG Chatbot Engine
- An asynchronous, multi-tenant Retrieval-Augmented Generation (RAG) chatbot engine custom-built for telecommunications help-desks. The system utilizes LangGraph for intent routing and self-correction loops, Google Gemini 2.5 Flash for streaming generation with native server-side context caching, Qdrant for payload-filtered metadata lookup isolation, and PostgreSQL for operational logging, customer profiling, and load-balanced live agent queue tracking.

🏗️ System Architecture & Data Flow

                    [User Request]
                               │
                               ▼
                       [FastAPI Router]
                               │
                               ▼
               [PostgreSQL Profile Lookup]
                               │
                               ▼
                    [Intent Classification]
                     /                 \
        (rag_lookup)/                   \(human_escalation)
                   v                     v
     [Qdrant Filtered Search]     [Capacity-Aware Queue Router]
                   │                     │ (FOR UPDATE SKIP LOCKED)
                   ▼                     ▼
        [Gemini Context Cache]    [Log Status: HUMAN_HANDOFF]
                   │                     │
                   ▼                     ▼
          [Verify Grounding]      [SSE Flag: human_handoff]

# Key Technical Pillars
1. Intelligent Entry Intent Gate: Analyzes inbound customer text prior to RAG injection using a Gemini structured output classification model. Explicit live agent demands or high-frustration indicators bypass vector parsing entirely to execute immediate transfers.
2. Concurrency-Safe Capacity Tracking: Uses an atomic row-level lock execution loop (FOR UPDATE SKIP LOCKED) inside PostgreSQL to instantly find available human support staff based on real-time capacity and matching customer tier SLAs (e.g., routing VIPs to enterprise_vip queues).
3. Multi-Tenant Data Isolation: Multi-tenancy metadata rules enforce separate document boundary parameters on the Qdrant cluster based on account profiles fetched during boot steps.
4. Gemini Native Context Caching: Automatically buffers background technical handbooks on remote Google Cloud infrastructure when retrieved data bounds clear 32,768 tokens, cutting latency and costs by up to 75%.

📁 Repository Directory Structure
telecom-rag-bot/
├── .github/workflows/         # Automated GitHub Actions CI/CD pipeline
├── config/                    # Global system setups & Pydantic settings parsing
├── database/                  # PostgreSQL pool managers, initialization vectors, & schemas
├── src/                       # Primary operational codebase workspace
│   ├── agents/                # LangGraph nodes, intent triage gates, and topologies
│   ├── api/                   # Transport gateway routers, pydantic schemas, and endpoints
│   ├── services/              # Third-party integrations (Qdrant, Cache Manager)
│   └── utils/                 # Structured JSON loggers and PII anonymizers
├── tests/                     # Automated validation, verification, and streaming test suites
├── Dockerfile                 # Slim multi-stage container configuration
├── docker-compose.yml         # Local environment stack composition orchestration
└── setup.py                   # Makes the project installable as an editable package

🚀 Local Quick-Start Workspace
Prerequisites
- Python 3.10, 3.11, or 3.12 installed locally.
- Docker and Docker Compose installed.
- An active Google Gemini API key.
  
1. Provision Environments and Secrets
 - Clone this repository to your system, then create a local environment variables profile in the project root directory:

# Initialize secrets repository configuration
cat <<EOF > .env
ENVIRONMENT=development
LOG_LEVEL=INFO
GEMINI_API_KEY=AIzaSyYourActualGeminiSecretAPIKeyHere
QDRANT_API_KEY=SecretQdrantClusterToken456!
QDRANT_URL=http://qdrant:6333
DB_CONNECTION_STRING=postgresql://telecom_admin:SecretProductionPassword123!@postgres:5432/telecom_db
EOF

2. Launch Local Datastore Infrastructures
- Spin up the complete system (PostgreSQL seeded with standard mock accounts, active live support agent pools, a Qdrant cluster initialized, and the FastAPI application gateway container) using a single command:

  docker compose up --build -d

Verify that all systems are operational and responding within normal bounds:

  docker compose ps
  docker compose logs -f api_gateway

🧪 Verification and Verification Testing
- The project includes unit, integration, and streaming performance assertion checks inside the tests/ directory.

Local Native Verification Running
- To isolate and run test modules directly on your local system path outside of container topologies, execute:
  
# Initialize virtual environment boundaries
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install workspace dependencies as an editable dev-packaged module
pip install -e .[dev]

# Run all test profiles using pytest
pytest tests/ -v --asyncio-mode=auto

📡 Live Stream API Validation Guide
- Once the docker container platform boots up, access interactive OpenAPI routing documentation via http://localhost:8000/docs (disabled inside explicit production environments).

Scenario A: Standard RAG Document Completion Request

curl -X POST "http://localhost:8000/v1/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST-1001",
       "question": "How do I configure my home router static APN parameters?",
       "session_id": "e3b4a5c6-7d8e-9f0a-1b2c-3d4e5f6a7b8c"
     }' \
     --no-buffer
     
Output Event Stream Wire Format (Standard):

 data: {"event": "metadata", "session_id": "e3b4a5c6-7d8e-9f0a-1b2c-3d4e5f6a7b8c"}

 data: {"token": "To ", "event_type": "token"}
 data: {"token": "configure ", "event_type": "token"}
 data: {"token": "your ", "event_type": "token"}
 data: {"token": "APN, ", "event_type": "token"}
 data: {"token": "navigate ", "event_type": "to..."}

 data: {"event": "completed"}

Scenario B: Triggering the Intent Gate Human Escalation Path

curl -X POST "http://localhost:8000/v1/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST-2002",
       "question": "Get me a human manager right now, this connection is totally broken!",
       "session_id": "e3b4a5c6-7d8e-9f0a-1b2c-3d4e5f6a7b8c"
     }' \
     --no-buffer
     
Output Event Stream Wire Format (Escalation):

 data: {"event": "metadata", "session_id": "e3b4a5c6-7d8e-9f0a-1b2c-3d4e5f6a7b8c"}

 data: {"event": "human_handoff", "session_id": "e3b4a5c6-7d8e-9f0a-1b2c-3d4e5f6a7b8c", "message": "I am transferring your request to our ENTERPRISE VIP live-agent support queue. An agent (ID: AGENT-503) will be visible in your chat window shortly."}

 data: {"event": "completed"}

🔒 Security & Data Masking Compliance
- This application guarantees strict compliance parameters for external LLM ingestion boundaries via the src/utils/anonymizer.py utility module.
   - Pre-Execution Scanning: Inbound user entries parse across specialized regular expression arrays to intercept and scrub Phone Numbers, Email Identities, IPv4/MAC configurations, and Credit Card lines.
   - Token Rehydration Vault: Sensitive fields are substituted with non-descriptive categorical symbols (e.g., [PHONE_NUMBER_1]) prior to network hops, and rehydrated locally right before rendering back to authenticated clients.

🛠️ Production Deployment Configuration Checklists
- When preparing this workspace image for Cloud environments (such as AWS EKS, Google GKE, or Azure AKS), verify the following orchestration parameters:
  - Nginx/Ingress Buffer Override: Ensure downstream proxy controllers pass headers un-buffered. Ingress routers must hold configuration snippets equivalent to:
    
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;
    proxy_buffering off;

- Persistent Cache Sweeper: Configure an automated Kubernetes CronJob to invoke cache_manager.purge_expired_caches() periodically to scrub stale server allocations that clear active TTL boundaries.
- Database Connection Limits: In highly distributed scaling clusters, ensure PgBouncer sits between the container pods and your primary master PostgreSQL database instance to handle the transactional FOR UPDATE query loops without pool exhaustion.
          
