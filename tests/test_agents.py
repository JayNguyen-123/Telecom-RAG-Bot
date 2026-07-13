# Graph routing verification
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langgraph.graph import END

# Absolute namespace imports enabled via setup.py local installation package rules
from src.agents.state import AgentState
from src.agents.workflow import TelecomGraphOrchestrator, GroundednessScore


@pytest.fixture
def mock_telecom_nodes():
    """Provides an isolated mock container mimicking underlying graph node outputs."""
    with patch("src.agents.nodes.telecom_nodes") as mock_nodes:
        # Configure baseline default behaviors for standard linear execution nodes
        mock_nodes.initialize_session.return_value = {
            "plan_tier": "enterprise",
            "region": "US-WEST",
            "account_status": "active",
            "metrics_log": [{"node": "initialize_session", "status": "SUCCESS"}]
        }
        mock_nodes.retrieve_context.return_value = {
            "documents": [Document(page_content="Enterprise 5G backup overrides standard fiber paths.")],
            "metrics_log": [{"node": "retrieve_context", "chunks_pulled": 1}]
        }
        mock_nodes.allocate_gemini_cache.return_value = {
            "cache_name": "mocked_google_cache_handle_hash",
            "metrics_log": [{"node": "allocate_gemini_cache", "cached": True}]
        }
        mock_nodes.generate_response_stub.return_value = {
            "generation": "Based on your Enterprise tier documentation, 5G backup overrides fiber paths automatically.",
            "metrics_log": [{"node": "generate_response_stub", "status": "COMPLETED"}]
        }
        yield mock_nodes


@pytest.fixture
def orchestrator_instance():
    """Instantiates a clean testing context instance for the state graph engine."""
    with patch("src.agents.workflow.ChatGoogleGenerativeAI") as mock_llm:
        # Prevent actual Gemini network requests during graph orchestration setups
        instance = TelecomGraphOrchestrator()
        yield instance


def test_graph_structural_topology_compilation(orchestrator_instance):
    """Verifies that the compiled LangGraph object contains all vital infrastructure nodes."""
    graph = orchestrator_instance.build_workflow()
    assert graph is not None
    
    # Inspect compiled graph node maps to verify all elements registered properly
    node_keys = graph.nodes.keys()
    assert "initialize_session" in node_keys
    assert "retrieve_context" in node_keys
    assert "allocate_gemini_cache" in node_keys
    assert "generate_response_stub" in node_keys


@pytest.mark.asyncio
async def test_linear_execution_flow_to_completion(orchestrator_instance, mock_telecom_nodes):
    """
    Triggers the compiled state machine and asserts that the state 
    propagates cleanly across all operational node steps.
    """
    with patch("src.agents.workflow.ChatGoogleGenerativeAI") as mock_llm:
        # Mock structural evaluator output to simulate a passing groundedness check
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = GroundednessScore(binary_score="yes")
        mock_llm.return_value.with_structured_output.return_value = mock_structured_llm

        graph = orchestrator_instance.build_workflow()
        
        initial_state = {
            "customer_id": "CUST-999",
            "session_id": "00000000-0000-0000-0000-000000000000",
            "question": "What happens when my fiber connection drops?",
            "documents": [],
            "plan_tier": "retail",
            "region": "GLOBAL",
            "account_status": "active",
            "cache_name": None,
            "generation": "",
            "metrics_log": []
        }

        # Execute the graph synchronously using an in-memory checkpointer context thread
        config = {"configurable": {"thread_id": "test-session"}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # Assert final mutated state matches expected outcomes from our node updates
        assert final_state["plan_tier"] == "enterprise"
        assert final_state["cache_name"] == "mocked_google_cache_handle_hash"
        assert "5G backup overrides fiber paths" in final_state["generation"]
        
        # Verify append reducer merged execution logs from separate nodes without data loss
        assert len(final_state["metrics_log"]) >= 4


def test_conditional_routing_on_hallucination(orchestrator_instance):
    """
    Asserts that the conditional routing edge catches a hallucinating response
    and accurately routes the state to the safety fallback pathway.
    """
    # Configure the mocked structured LLM to flag a hallucination violation
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = GroundednessScore(binary_score="no")
    orchestrator_instance.evaluator_llm = mock_structured_llm

    mock_state: AgentState = {
        "customer_id": "CUST-000",
        "session_id": "11111111-1111-1111-1111-111111111111",
        "question": "Is international roaming free?",
        "documents": [Document(page_content="Roaming is billed at $10/day globally.")],
        "plan_tier": "retail",
        "region": "GLOBAL",
        "account_status": "active",
        "cache_name": None,
        "generation": "Yes, roaming is completely free worldwide!", # Deliberate hallucination
        "metrics_log": []
    }

    # Manually execute the graph's conditional edge logic function
    routing_decision = orchestrator_instance.verify_grounding_safety(mock_state)
    
    # Assert that the system accurately routed the session directly to the safety fallback endpoint
    assert routing_decision == "fallback"
