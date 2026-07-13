# Qdrant filtering validation
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from qdrant_client.http import models as qmodels

# Absolute imports enabled via local editable package configuration rules
from src.agents.state import AgentState
from src.agents.nodes import telecom_nodes


@pytest.fixture
def mock_vector_store():
    """Patches the get_qdrant_vector_store provider dependency inside nodes layer."""
    with patch("src.agents.nodes.get_qdrant_vector_store") as mock_store_provider:
        mock_store = MagicMock()
        mock_store_provider.return_value = mock_store
        yield mock_store


def test_retrieve_context_enforces_metadata_isolation(mock_vector_store):
    """
    Verifies that search requests map the customer's plan_tier payload 
    directly into Qdrant's structural filtering query parameters.
    """
    # 1. Arrange: Setup simulated vector return payloads
    expected_docs = [
        Document(page_content="Enterprise 5G dedicated failover architecture specifications.", metadata={"plan_tier": "enterprise"})
    ]
    mock_vector_store.similarity_search.return_value = expected_docs

    test_state: AgentState = {
        "customer_id": "CUST-2002",
        "session_id": "44444444-4444-4444-4444-444444444444",
        "question": "How do I configure my primary corporate fallback routing?",
        "documents": [],
        "plan_tier": "enterprise", # Target value to filter on
        "region": "US-WEST",
        "account_status": "active",
        "cache_name": None,
        "generation": "",
        "metrics_log": []
    }

    # 2. Act: Trigger execution node
    updated_state = telecom_nodes.retrieve_context(test_state)

    # 3. Assert: Verify the vector storage layer received the correct structural filter parameters
    mock_vector_store.similarity_search.assert_called_once()
    
    # Extract structural arguments sent to Qdrant search methods
    called_kwargs = mock_vector_store.similarity_search.call_args[1]
    
    assert called_kwargs["query"] == test_state["question"]
    assert called_kwargs["k"] == 6
    
    # Assert filter payload matches multi-tenant strict compliance specs
    qdrant_filter = called_kwargs["filter"]
    assert isinstance(qdrant_filter, qmodels.Filter)
    assert len(qdrant_filter.must) == 1
    
    condition = qdrant_filter.must[0]
    assert condition.key == "metadata.plan_tier"
    assert condition.match.value == "enterprise"
    
    # Assert returned state contains the isolated document payloads
    assert updated_state["documents"] == expected_docs
    assert updated_state["metrics_log"][0]["chunks_pulled"] == 1


def test_retrieve_context_graceful_empty_fallback(mock_vector_store):
    """
    Ensures that if the vector index experiences connection hiccups or returns 
    no documents, the node fails gracefully instead of crashing the state machine.
    """
    # Configure vector store exception simulation
    mock_vector_store.similarity_search.side_effect = Exception("Connection timed out to Qdrant cluster cluster.")

    test_state: AgentState = {
        "customer_id": "CUST-1001",
        "session_id": "55555555-5555-5555-5555-555555555555",
        "question": "What is my dynamic IP address allocation?",
        "documents": [],
        "plan_tier": "retail",
        "region": "GLOBAL",
        "account_status": "active",
        "cache_name": None,
        "generation": "",
        "metrics_log": []
    }

    # Execute step node safely
    updated_state = telecom_nodes.retrieve_context(test_state)

    # Assert state boundaries did not dissolve or trigger termination conditions
    assert updated_state["documents"] == []
    assert updated_state["metrics_log"][0]["status"] == "RETRIEVAL_ERROR"
