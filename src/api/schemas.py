# Inbound/outbound payload validation models
import uuid
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    """
    Validates the inbound payload required to initialize a chat session stream.
    """
    customer_id: str = Field(
        ...,
        description="Unique system identifier for the customer, used to look up account tier rules.",
        examples=["CUST-1001", "CUST-2002"]
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="The customer support question or message query.",
        examples=["How do I set up 5G failover on my router?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional tracking identifier. If left empty, a fresh UUID token will be provisioned.",
        examples=["e3b4a5c6-7d8e-9f0a-1b2c-3d4e5f6a7b8c"]
    )

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, value: str) -> str:
        """Strip raw spacing and enforce parameter completeness checks."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("customer_id cannot consist solely of empty text spacing characters.")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: Optional[str]) -> Optional[str]:
        """Verify that incoming session strings conform to valid structural UUID specs if present."""
        if value is None:
            return value
        try:
            uuid.UUID(value)
            return value
        except ValueError as e:
            raise ValueError("Provided session_id parameter string must follow standard UUID format specs.") from e


class ChatResponseStreamChunk(BaseModel):
    """
    Structural validation model wrapper for outgoing Server-Sent Event (SSE) stream tokens.
    """
    token: str = Field(
        ...,
        description="The single text fragment or structural token emitted by the generative engine."
    )
    event_type: Literal["token", "metadata", "error", "heartbeat"] = Field(
        default="token",
        description="Categorization label allowing the receiving client dashboard UI hook to parse routing rules."
    )
