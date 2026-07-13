# Step definitions (Retrieve, Stream, Cache)
import logging
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.http import models as qmodels
from google import genai
from google.genai import types

from config.settings import settings
from database.connection import db_manager
from src.agents.state import AgentState
from src.services.qdrant_client import get_qdrant_vector_store

logger = logging.getLogger(__name__)

# --- Structured Output Schema for Intent Classification Gate ---
class IntentClassification(BaseModel):
    """Structured response schema used to route user intent at the entry point."""
    intent: Literal["rag_lookup", "human_escalation"] = Field(
        description="Categorize if the user wants to talk to a human agent/expresses critical frustration or if it can be resolved via knowledge lookup."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class TelecomGraphNodes:
    """
    Implements operational processing steps used within the LangGraph topology.
    Encapsulates backend services, database pooling, and generative runtimes.
    """
    def __init__(self):
        self.vector_store = get_qdrant_vector_store()
        self.evaluator_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0
        )
        self.intent_classifier = self.evaluator_llm.with_structured_output(IntentClassification)
        self.google_genai_client = genai.Client()

    def initialize_session(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 1: Pulls user account telemetry details from the PostgreSQL instance.
        """
        customer_id = state["customer_id"]
        logger.info(f"Executing initialize_session node for customer: {customer_id}")

        query = """
            SELECT plan_tier, region, account_status
            FROM telecom_rag.customer_profiles
            WHERE customer_id = %s;
        """

        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute(query, (customer_id,))
                row = cursor.fetchone()

            if not row:
                logger.warning(f"Customer identifier {customer_id} not located. Defaulting to retail profile tier.")
                return {
                    "plan_tier": "retail",
                    "region": "GLOBAL",
                    "account_status": "active",
                    "metrics_log": [{"node": "initialize_session", "status": "CUSTOMER_NOT_FOUND"}]
                }

            return {
                "plan_tier": row[0],
                "region": row[1],
                "account_status": row[2],
                "metrics_log": [{"node": "initialize_session", "status": "SUCCESS"}]
            }
        except Exception as e:
            logger.error(f"PostgreSQL context lookup failed inside initialize_session node: {e}")
            raise RuntimeError("Database connectivity fault within agent workflow.") from e

    def route_intent(self, state: AgentState) -> str:
        """
        Conditional Entry Gate: Evaluates if the user query requires direct human escalation,
        bypassing the retrieval layers completely.
        """
        logger.info(f"Classifying user intent for question: {state['question']}")

        prompt = ChatPromptTemplate.from_template(
            "Analyze the user conversation query below.\n"
            "If the user is explicitly demanding a human, live agent, support representative, "
            "or expressing severe frustration that requires a human manager, classify as 'human_escalation'.\n"
            "Otherwise, choose 'rag_lookup'.\n\n"
            "User Query: {question}"
        )

        try:
            result = self.intent_classifier.invoke({"question": state["question"]})
            if result.intent == "human_escalation":
                logger.warning(f"Live human agent request caught with confidence {result.confidence}. Routing to handoff.")
                return "handle_human_handoff"
            return "retrieve_context"
        except Exception as e:
            logger.error(f"Intent classification failed, defaulting to standard RAG pipeline. Error: {e}")
            return "retrieve_context"

    def retrieve_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 2: Executes a metadata-filtered vector search against the Qdrant cluster.
        """
        logger.info(f"Executing retrieve_context node for session: {state['session_id']}")

        qdrant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.plan_tier",
                    match=qmodels.MatchValue(value=state["plan_tier"])
                )
            ]
        )

        try:
            docs = self.vector_store.similarity_search(
                query=state["question"],
                k=6,
                filter=qdrant_filter
            )
            return {
                "documents": docs,
                "metrics_log": [{"node": "retrieve_context", "chunks_pulled": len(docs)}]
            }
        except Exception as e:
            logger.error(f"Vector search retrieval sequence broke inside Qdrant: {e}")
            return {
                "documents": [],
                "metrics_log": [{"node": "retrieve_context", "status": "RETRIEVAL_ERROR"}]
            }

    def allocate_gemini_cache(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 3: Dynamically provisions server-side token caches if context bounds scale.
        """
        docs = state.get("documents", [])
        if not docs:
            return {"cache_name": None}

        combined_contents = "\n\n=== REFERENCE SEGMENT ===\n\n".join([d.page_content for d in docs])
        estimated_token_count = len(combined_contents) // 4

        if estimated_token_count < settings.MIN_TOKENS_FOR_CACHING:
            return {
                "cache_name": None,
                "metrics_log": [{"node": "allocate_gemini_cache", "cached": False, "estimated_tokens": estimated_token_count}]
            }

        try:
            cache = self.google_genai_client.caches.create(
                model=settings.LLM_MODEL_NAME,
                config=types.CreateCachedContentConfig(
                    contents=[combined_contents],
                    display_name=f"session_cache_{state['session_id']}",
                    ttl=f"{settings.CONTEXT_CACHE_TTL_SECONDS}s",
                )
            )
            return {
                "cache_name": cache.name,
                "metrics_log": [{"node": "allocate_gemini_cache", "cached": True, "cache_name": cache.name}]
            }
        except Exception as e:
            logger.error(f"Failed to establish Gemini server-side context content cache: {e}")
            return {
                "cache_name": None,
                "metrics_log": [{"node": "allocate_gemini_cache", "status": "CACHE_ALLOCATION_FAILURE"}]
            }

    def generate_response_stub(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 4: Non-streaming completion node used to verify answer grounding.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are a Telecom Customer Support Expert.\n"
            "Answer the query using ONLY the verified facts attached.\n"
            "If the answer cannot be confidently formulated, state that you cannot assist.\n\n"
            "Context Source Material:\n{context}\n\n"
            "Customer Query: {question}"
        )

        try:
            if state.get("cache_name"):
                llm_engine = ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL_NAME,
                    temperature=0.1,
                    extra_google_params={"cached_content": state["cache_name"]}
                )
                chain = prompt | llm_engine
                response = chain.invoke({"context": "[Using Cached Mapping]", "question": state["question"]})
            else:
                chain = prompt | self.evaluator_llm
                context_str = "\n\n".join([d.page_content for d in state["documents"]])
                response = chain.invoke({"context": context_str, "question": state["question"]})

            return {
                "generation": response.content,
                "metrics_log": [{"node": "generate_response_stub", "status": "COMPLETED"}]
            }
        except Exception as e:
            logger.error(f"Response execution block failed: {e}")
            return {
                "generation": "I am experiencing technical difficulties processing this support query.",
                "metrics_log": [{"node": "generate_response_stub", "status": "GENERATION_FAILURE"}]
            }

    def handle_human_handoff(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 5: Dynamic Capacity-Aware Handoff Node. Triage matching account configurations,
        queries PostgreSQL for an available agent within the queue, increments occupancy metrics,
        and flags the interaction logs.
        """
        session_id = state["session_id"]
        customer_id = state["customer_id"]
        plan_tier = state.get("plan_tier", "retail")

        # Route to specialized queues depending directly on customer plan tiers
        target_queue = "enterprise_vip" if plan_tier in ["enterprise", "vip"] else "technical"
        logger.warning(f"Initiating live agent handoff sequence to queue: {target_queue}")

        find_agent_query = """
            SELECT agent_id FROM telecom_rag.human_agent_registry
            WHERE assigned_queue = %s
              AND agent_status = 'available'
              AND current_capacity < max_capacity
            ORDER BY (max_capacity - current_capacity) DESC
            FOR UPDATE SKIP LOCKED
            LIMIT 1;
        """

        assign_agent_query = """
            UPDATE telecom_rag.human_agent_registry
            SET
                current_capacity = current_capacity + 1,
                agent_status = CASE
                                WHEN current_capacity + 1 >= max_capacity THEN 'busy'
                                ELSE 'available'
                               END
            WHERE agent_id = %s;
        """

        log_escalation_query = """
            INSERT INTO telecom_rag.rag_interaction_logs(
                session_id, customer_id, user_question, llm_generation,
                execution_status, escalated_to_queue, assigned_agent_id, escalation_timestamp
            )
            VALUES (%s, %s, %s, %s, 'HUMAN_HANDOFF', %s, %s, CURRENT_TIMESTAMP);
        """

        assigned_agent_id = None
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Look up the most available agent using a row-level lock (FOR UPDATE SKIP LOCKED)
                    cursor.execute(find_agent_query, (target_queue,))
                    row = cursor.fetchone()

                    if row:
                        assigned_agent_id = row[0]
                        # 2. Increment active agent concurrency bounds safely inside the transaction
                        cursor.execute(assign_agent_query, (assigned_agent_id,))
                        handoff_msg = (
                            f"I am transferring your request to our {target_queue.replace('_', ' ').upper()} live-agent support queue. "
                            f"An agent (ID: {assigned_agent_id if assigned_agent_id else 'QUEUE_POOL'}) will be visible in your chat window shortly."
                        )
                        # 3. Insert operational audit trail log marking routing coordinates
                        cursor.execute(log_escalation_query, (session_id, customer_id, state["question"], handoff_msg, target_queue, assigned_agent_id))

                        return {
                            "generation": handoff_msg,
                            "metrics_log": [{"node": "handle_human_handoff", "status": "ESCALATED", "agent_assigned": assigned_agent_id}]
                        }
                    else:
                        # No agents available
                        handoff_msg = "All representative queues are full. Please try your request again shortly."
                        # Still log the escalation attempt even if no agent was assigned
                        cursor.execute(log_escalation_query, (session_id, customer_id, state["question"], handoff_msg, target_queue, None))
                        return {
                            "generation": handoff_msg,
                            "metrics_log": [{"node": "handle_human_handoff", "status": "NO_AGENT_AVAILABLE"}]
                        }

        except Exception as e:
            logger.error(f"Transactional routing failure executing human agent allocation pool adjustments: {e}")
            return {
                "generation": "I am experiencing technical difficulties processing your request for a human agent. Please try again later.",
                "metrics_log": [{"node": "handle_human_handoff", "status": "DATABASE_TRANSACTION_ERROR"}]
            }

# Global node instantiation handle
telecom_nodes = TelecomGraphNodes()
