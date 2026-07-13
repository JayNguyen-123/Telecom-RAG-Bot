# FastAPI routing and SSE streaming channels
import json
import logging
import uuid
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, APIRouter, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from config.settings import settings
from database.connection import db_manager
from src.api.schemas import ChatRequest, ChatResponseStreamChunk
from src.agents.workflow import rag_graph

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events cleanly.
    Ensures infrastructure pools initialize and terminate correctly.
    """
    logger.info("Starting up API service dependencies...")
    try:
        db_manager.initialize_pool()
        logger.info("Database connection pool ready.")
    except Exception as e:
        logger.critical(f"Failed to initialize database pool during startup: {e}")
        raise SystemExit(1)
        
    yield
    
    logger.info("Shutting down API service dependencies...")
    try:
        db_manager.close_all_connections()
        logger.info("Database connection pool closed successfully.")
    except Exception as e:
        logger.error(f"Error closing database pool during shutdown: {e}")

# Instantiate main FastAPI application core
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# Apply restrictive CORS policies for cross-origin compliance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT != "production" else ["https://*.telecom-corp.internal"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

router = APIRouter(prefix="/v1/chat", tags=["AI Chat Execution Engines"])

async def execute_graph_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Executes the compiled LangGraph orchestration state machine.
    Detects if an escalation node triggered, formats specialized structural 
    JSON frames, or processes regular RAG semantic token streaming.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Initiating graph stream context loop for session: {session_id}")
    
    # Establish base tracking state matching the AgentState schema definition
    initial_state = {
        "customer_id": request.customer_id,
        "session_id": session_id,
        "question": request.question,
        "documents": [],
        "plan_tier": "retail",
        "region": "GLOBAL",
        "account_status": "active",
        "cache_name": None,
        "generation": "",
        "metrics_log": []
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Step 1: Send initialization metadata frame directly to consumer client hooks
        yield f"data: {json.dumps({'event': 'metadata', 'session_id': session_id})}\n\n"
        await asyncio.sleep(0.01) # Yield execution window momentarily to event loop
        
        # Step 2: Execute the full state graph process tree synchronously 
        final_output = await rag_graph.ainvoke(initial_state, config=config)
        
        # Step 3: Parse the append-only logs tracker to verify if a human handoff event occurred
        is_human_escalation = any(
            metric.get("node") == "handle_human_handoff" and metric.get("status") == "ESCALATED"
            for metric in final_output.get("metrics_log", [])
        )
        
        if is_human_escalation:
            # Emit a specialized telemetry block allowing UI layer to switch to a live desk state instantly
            yield f"data: {json.dumps({'event': 'human_handoff', 'session_id': session_id, 'message': final_output['generation']})}\n\n"
            yield f"data: {json.dumps({'event': 'completed'})}\n\n"
            return

        # Step 4: If not an escalation, split generation text string into fragments to simulate token streaming
        # In a fully streaming setup, you can loop directly across your graph's `astream()` events chunk-by-chunk.
        generated_text = final_output.get("generation", "")
        words = generated_text.split(" ")
        
        for index, word in enumerate(words):
            # Append trailing space characters back to word strings except the final element
            token_fragment = word if index == len(words) - 1 else f"{word} "
            chunk_payload = ChatResponseStreamChunk(token=token_fragment, event_type="token")
            
            yield f"data: {chunk_payload.model_dump_json()}\n\n"
            await asyncio.sleep(0.02) # Pacing latency emulator for smoother UI rendering
            
        # Step 5: Push final completion marker signal 
        yield f"data: {json.dumps({'event': 'completed'})}\n\n"
        
    except asyncio.CancelledError:
        logger.warning(f"Client terminated websocket or HTTP connection during streaming loop: {session_id}")
        raise
    except Exception as e:
        logger.error(f"Unexpected internal failure within API streaming loop block: {e}")
        yield f"data: {json.dumps({'event': 'error', 'message': 'Internal processing server error occurred processing request'})}\n\n"


@router.post("/stream", summary="Initiates a live-token SSE stream supporting automated human handoffs")
async def chat_stream_endpoint(request: ChatRequest):
    """
    FastAPI Router entry point handling real-time token streaming channels.
    Accepts validated schema models and wraps the response inside an explicit Event-Stream wrapper.
    """
    try:
        if not request.customer_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Customer ID identifier parameter cannot be empty."
            )
            
        return StreamingResponse(
            execute_graph_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no" # Prevents Nginx/proxy buffer layers from stalling tokens
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to construct target streaming channel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while establishing the real-time processing interface."
        )

app.include_router(router)

def main():
    """Entry point definition linked to setup.py console_scripts configuration hooks."""
    import uvicorn
    uvicorn.run("src.api.endpoints:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
