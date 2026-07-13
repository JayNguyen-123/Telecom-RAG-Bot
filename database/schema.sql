# SQL definitions for chat logs and customer profiles
-- Production Database Schema Initialization Script
-- Target: PostgreSQL 14+

CREATE SCHEMA IF NOT EXISTS telecom_rag;

-- Set local execution context to target schema
SET search_path TO telecom_rag, public;

-----------------------------------------
-- TABLE: CUSTOMER PROFILES
-----------------------------------------
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id VARCHAR(64) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    plan_tier VARCHAR(32) NOT NULL DEFAULT 'retail',
    region VARCHAR(50) NOT NULL,
    account_status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure explicit structural tiers
    CONSTRAINT chk_plan_tier CHECK (plan_tier IN ('retail', 'business', 'enterprise', 'vip')),
    CONSTRAINT chk_account_status CHECK (account_status IN ('active', 'suspended', 'terminated'))
);

-- Performance index for account isolation sorting
CREATE INDEX IF NOT EXISTS idx_customer_profiles_tier_status 
ON customer_profiles(plan_tier, account_status);

-----------------------------------------
-- TABLE: HUMAN AGENT REGISTRY
-----------------------------------------
CREATE TABLE IF NOT EXISTS human_agent_registry (
    agent_id VARCHAR(64) PRIMARY KEY,
    full_name VARCHAR(200) NOT NULL,
    assigned_queue VARCHAR(50) NOT NULL, -- e.g., 'technical', 'billing', 'enterprise_vip'
    agent_status VARCHAR(20) NOT NULL DEFAULT 'available',
    current_capacity INT NOT NULL DEFAULT 0,
    max_capacity INT NOT NULL DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_agent_status CHECK (agent_status IN ('available', 'busy', 'offline', 'break'))
);

-- Performance index for agent assignment matching
CREATE INDEX IF NOT EXISTS idx_agents_routing_lookup 
ON human_agent_registry(assigned_queue, agent_status, current_capacity);

-----------------------------------------
-- TABLE: RAG INTERACTION LOGS & HUMAN ESCALATIONS
-----------------------------------------
CREATE TABLE IF NOT EXISTS rag_interaction_logs (
    interaction_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    customer_id VARCHAR(64) REFERENCES customer_profiles(customer_id) ON DELETE SET NULL,
    user_question TEXT NOT NULL,
    llm_generation TEXT,
    execution_status VARCHAR(32) NOT NULL,
    response_latency_ms INT,
    
    -- Telemetry metrics for evaluating Gemini Native Context Caching efficiency
    cache_utilized BOOLEAN DEFAULT FALSE,
    cache_token_count INT DEFAULT 0,
    gemini_cache_name VARCHAR(255),
    
    -- Human Handoff Routing Integration Elements
    escalated_to_queue VARCHAR(50),
    assigned_agent_id VARCHAR(64) REFERENCES human_agent_registry(agent_id) ON DELETE SET NULL,
    escalation_timestamp TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Enforce explicit taxonomy checks across agent state loops
    CONSTRAINT chk_execution_status CHECK (execution_status IN (
        'SUCCESS', 
        'BLOCKED_HALLUCINATION', 
        'BLOCKED_SAFETY', 
        'SYSTEM_ERROR', 
        'FALLBACK', 
        'HUMAN_HANDOFF'
    ))
);

-- Indexing strategies for session lookups and dashboard analytics reporting
CREATE INDEX IF NOT EXISTS idx_rag_logs_session_id ON rag_interaction_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_rag_logs_customer_id ON rag_interaction_logs(customer_id);
CREATE INDEX IF NOT EXISTS idx_rag_logs_status_escalation ON rag_interaction_logs(execution_status, escalated_to_queue);
CREATE INDEX IF NOT EXISTS idx_rag_logs_created_at ON rag_interaction_logs(created_at DESC);

-----------------------------------------
-- SEED DATA FOR SANDBOX VALIDATION
-----------------------------------------
-- Seed Customer Profiles
INSERT INTO customer_profiles (customer_id, first_name, last_name, plan_tier, region, account_status)
VALUES 
('CUST-1001', 'Jane', 'Doe', 'retail', 'US-NORTHEAST', 'active'),
('CUST-2002', 'Acme', 'Telecom-Corp', 'enterprise', 'US-WEST', 'active'),
('CUST-3003', 'John', 'Smith', 'vip', 'US-SOUTH', 'suspended')
ON CONFLICT (customer_id) DO NOTHING;

-- Seed Live Human Support Agents
INSERT INTO human_agent_registry (agent_id, full_name, assigned_queue, agent_status, max_capacity)
VALUES 
('AGENT-501', 'Alex Mercer', 'technical', 'available', 4),
('AGENT-502', 'Sarah Jenkins', 'billing', 'available', 5),
('AGENT-503', 'David Vance', 'enterprise_vip', 'busy', 3)
ON CONFLICT (agent_id) DO NOTHING;
