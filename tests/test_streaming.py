# FastAPI endpoint event stream assertion
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Absolute imports enabled via local editable package configuration rules
from src.api.endpoints import app


@pytest.fixture
def api_client():
    """Provides a thread-safe, isolated FastAPI infrastructure test client instance."""
    return TestClient(app)


def test_chat_streaming_endpoint_headers(api_client):
    """
    Asserts that the endpoint responds with appropriate event-stream headers
    to prevent proxy layers like Nginx or Cloudflare from buffering real-time tokens.
    """
    payload = {
        "customer_id": "CUST-1001",
        "question": "How do I configure my home router's static APN?",
        "session_id": "99999999-9999-9999-9999-999999999999"
    }

    # Simulate standard non-escalation flow by mocking graph output
    with patch("src.api.endpoints.rag_graph.ainvoke") as mock_invoke:
        mock_invoke.return_value = {
            "generation": "To configure a static APN, navigate to your router setup page...",
            "metrics_log": [{"node": "generate_response_stub", "status": "COMPLETED"}]
        }

        # Use a context manager over the request stream to intercept headers instantly
        with api_client.stream("POST", "/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            
            # Validate mission-critical production proxy configuration headers
            assert response.headers["content-type"] == "text/event-stream"
            assert response.headers["cache-control"] == "no-cache, no-transform"
            assert response.headers["connection"] == "keep-alive"
            assert response.headers["x-accel-buffering"] == "no"


def test_chat_streaming_standard_rag_sequence(api_client):
    """
    Streams and parses a standard RAG response to ensure token fragments
    are sent properly and conform to the Pydantic schema contract.
    """
    payload = {
        "customer_id": "CUST-2002",
        "question": "What is the automated SLA failover resolution threshold?",
        "session_id": None  # Trigger automatic UUID generation check
    }

    with patch("src.api.endpoints.rag_graph.ainvoke") as mock_invoke:
        mock_invoke.return_value = {
            "generation": "The SLA failover activation threshold is 15 seconds.",
            "metrics_log": [
                {"node": "initialize_session", "status": "SUCCESS"},
                {"node": "retrieve_context", "chunks_pulled": 2},
                {"node": "generate_response_stub", "status": "COMPLETED"}
            ]
        }

        with api_client.stream("POST", "/v1/chat/stream", json=payload) as response:
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    json_str = line.replace("data: ", "").strip()
                    events.append(json.loads(json_str))

            assert len(events) > 0
            
            # 1. Assert Metadata Frame Event Accuracy
            assert events[0]["event"] == "metadata"
            assert "session_id" in events[0]

            # 2. Assert Live Token Generation Stream Behavior
            token_events = [ev for ev in events if ev.get("event_type") == "token"]
            assert len(token_events) > 0
            assert "token" in token_events[0]
            
            # 3. Assert Completion Sentinel Sequence
            assert events[-1]["event"] == "completed"


def test_chat_streaming_human_handoff_sequence(api_client):
    """
    Asserts that if the LangGraph state machine triggers the human escalation path,
    the streaming endpoint intercepts it and surfaces a structured 'human_handoff' event.
    """
    payload = {
        "customer_id": "CUST-2002",
        "question": "Connect me to a manager right now!",
        "session_id": "88888888-8888-8888-8888-888888888888"
    }

    with patch("src.api.endpoints.rag_graph.ainvoke") as mock_invoke:
        # Emulate the final state returned by the graph following a human handoff
        mock_invoke.return_value = {
            "generation": "I am transferring your request to our ENTERPRISE VIP live-agent support queue...",
            "metrics_log": [
                {"node": "initialize_session", "status": "SUCCESS"},
                {"node": "handle_human_handoff", "status": "ESCALATED", "agent_assigned": "AGENT-503"}
            ]
        }

        with api_client.stream("POST", "/v1/chat/stream", json=payload) as response:
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    json_str = line.replace("data: ", "").strip()
                    events.append(json.loads(json_str))

            assert len(events) == 3 # Metadata, Human Handoff Event, and Completed Sentinel
            
            # Verify the initialization metadata event frame
            assert events[0]["event"] == "metadata"
            
            # Verify the specialized structural human handoff event frame
            assert events[1]["event"] == "human_handoff"
            assert "ENTERPRISE VIP" in events[1]["message"]
            assert events[1]["session_id"] == payload["session_id"]
            
            # Ensure standard text chunk fields are omitted to avoid UI state confusion
            assert "event_type" not in events[1]
            
            # Verify orderly termination frame
            assert events[2]["event"] == "completed"


def test_chat_streaming_validation_boundaries(api_client):
    """
    Ensures that empty parameter inputs are rejected at the HTTP gateway boundary,
    preventing bad strings from running up processing fees within the LangGraph loops.
    """
    bad_payload = {
        "customer_id": "   ",  # Invalid blank spacing payload
        "question": "Short",
        "session_id": "malformed-non-uuid-string"
    }

    response = api_client.post("/v1/chat/stream", json=bad_payload)
    
    assert response.status_code == 422  # Unprocessable Entity
    error_details = response.json()["detail"]
    
    assert any("customer_id" in err["loc"] for err in error_details)
    assert any("session_id" in err["loc"] for err in error_details)
