# Conditional logic compilation graph
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from src.agents.state import AgentState
from src.agents.nodes import telecom_nodes

logger = logging.getLogger(__name__)

# --- Structured Output Schema for Validation ---
class GroundednessScore(BaseModel):
    """Structured response schema used to check for model hallucinations."""
    binary_score: str = Field(
        description="Is the generated answer strictly grounded in the given context facts? 'yes' or 'no'",
        pattern="^(yes|no)$"
    )

class TelecomGraphOrchestrator:
    """
    Assembles, links, and compiles the LangGraph State Machine.
    Manages the structural topology, roots early entry intents, 
    and routes states based on safety evaluations.
    """
    def __init__(self):
        # Initialize an evaluation model specifically for evaluating grounding metrics
        self.evaluator_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME, 
            temperature=0
        ).with_structured_output(GroundednessScore)

    def verify_grounding_safety(self, state: AgentState) -> str:
        """
        Conditional Router: Evaluates whether the generated answer matches the source documents.
        If a hallucination is detected, it reroutes the state to a fallback step to prevent bad info.
        """
        logger.info("Running post-generation grounding evaluation...")
        
        # If no generation took place due to upstream failures, skip evaluation and exit
        if not state.get("generation") or not state.get("documents"):
            return "fallback"
            
        prompt = ChatPromptTemplate.from_template(
            "Verify if the proposed response hallucinates metrics outside the provided facts.\n"
            "Context Material:\n{context}\n\n"
            "Proposed Response:\n{generation}\n\n"
            "Output your strict evaluation score."
        )
        
        context_str = "\n\n".join([d.page_content for d in state["documents"]])
        
        try:
            # Use structured outputs to get a reliable 'yes' or 'no' response
            result = self.evaluator_llm.invoke({
                "context": context_str, 
                "generation": state["generation"]
            })
            
            if result.binary_score == "yes":
                logger.info("Grounding verification passed. Finalizing graph transaction.")
                return "finalize"
                
            logger.warning("Hallucination detected in response chunk. Rerouting to safety fallback.")
            return "fallback"
            
        except Exception as e:
            logger.error(f"Failed to execute grounding check workflow: {e}")
            return "fallback"

    def build_workflow(self) -> StateGraph:
        """
        Builds and wires the nodes together into an operational graph topology.
        Integrates early intent triage and conditional execution routing.
        """
        # 1. Initialize Graph with state schemas definitions
        builder = StateGraph(AgentState)
        
        # 2. Register functional execution steps
        builder.add_node("initialize_session", telecom_nodes.initialize_session)
        builder.add_node("retrieve_context", telecom_nodes.retrieve_context)
        builder.add_node("allocate_gemini_cache", telecom_nodes.allocate_gemini_cache)
        builder.add_node("generate_response_stub", telecom_nodes.generate_response_stub)
        builder.add_node("handle_human_handoff", telecom_nodes.handle_human_handoff)
        
        # 3. Establish initial conditional routing gate immediately after boot
        builder.set_entry_point("initialize_session")
        
        builder.add_conditional_edges(
            "initialize_session",
            telecom_nodes.route_intent,
            {
                "retrieve_context": "retrieve_context",
                "handle_human_handoff": "handle_human_handoff"
            }
        )
        
        # 4. Map the sequential execution trail for the standard RAG pipeline
        builder.add_edge("retrieve_context", "allocate_gemini_cache")
        builder.add_edge("allocate_gemini_cache", "generate_response_stub")
        
        # 5. Direct terminal lines out from human handoff execution node
        builder.add_edge("handle_human_handoff", END)
        
        # 6. Insert conditional routing gates to protect against bad outputs
        builder.add_conditional_edges(
            "generate_response_stub",
            self.verify_grounding_safety,
            {
                "finalize": END,
                "fallback": END  # In production, route this to an isolation fallback message or custom retry loop
            }
        )
        
        # 7. Compile with an in-memory checkpointer to preserve chat state across steps
        memory_checkpoint = MemorySaver()
        compiled_graph = builder.compile(checkpointer=memory_checkpoint)
        logger.info("LangGraph workflow compiled successfully with Intent and Escalation paths.")
        return compiled_graph

# Instantiate global orchestrator singleton
orchestrator = TelecomGraphOrchestrator()
rag_graph = orchestrator.build_workflow()
