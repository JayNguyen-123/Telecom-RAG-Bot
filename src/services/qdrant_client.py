# High-speed semantic search and payload filtering
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenAIEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

class QdrantInfrastructureManager:
    """
    Manages connections and collection lifecycle operations for the Qdrant vector database.
    Implements a singleton pattern to safely reuse the client across runtime components.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantInfrastructureManager, cls).__new__(cls)
            cls._instance._client = None
            cls._instance._vector_store = None
        return cls._instance

    def get_client(self) -> QdrantClient:
        """Returns or instantiates a raw Qdrant client connection instance."""
        if self._client is None:
            logger.info(f"Establishing active connection to Qdrant cluster at {settings.QDRANT_URL}")
            # Secure key ingestion from settings layer
            api_key = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
            
            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=api_key,
                timeout=10.0 # Prevents request blocking loops
            )
        return self._client

    def bootstrap_collection(self) -> None:
        """
        Idempotent database setup routine. Verifies collection existence, provisions
        HNSW optimization profiles, and creates standard structural payload index paths.
        """
        client = self.get_client()
        collection_name = settings.QDRANT_COLLECTION_NAME

        try:
            # Check if the target collection already exists
            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists:
                logger.info(f"Target Qdrant collection '{collection_name}' already initialized.")
                return

            logger.warning(f"Collection '{collection_name}' not found. Initializing storage parameters...")
            
            # Create collection using specific technical parameters matching Gemini embeddings
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.VECTOR_DIMENSION, # Typically 768 dimensions for models/text-embedding-004
                    distance=qmodels.Distance.COSINE,
                    on_disk=True # Offloads index to disk storage to minimize production memory footprints
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    on_disk=True
                )
            )

            # Create payload index for performance isolation on plan_tier values
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.plan_tier",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Successfully bootstrapped collection '{collection_name}' with optimized keyword indexes.")

        except Exception as e:
            logger.critical(f"Critical initialization failure provisioning Qdrant datastore layers: {e}")
            raise RuntimeError("Could not construct infrastructure vector boundaries.") from e

    def get_vector_store(self) -> QdrantVectorStore:
        """
        Wraps the verified connection into a unified LangChain operational interface object.
        """
        if self._vector_store is None:
            # Ensure index infrastructure exists before generating abstract wrappers
            self.bootstrap_collection()
            
            embeddings = GoogleGenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
            
            self._vector_store = QdrantVectorStore(
                client=self.get_client(),
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embeddings=embeddings
            )
        return self._vector_store

# Instantiate global service interface handlers
qdrant_manager = QdrantInfrastructureManager()

def get_qdrant_vector_store() -> QdrantVectorStore:
    """Convenience dependency injector function referenced inside src.agents.nodes."""
    return qdrant_manager.get_vector_store()

def main():
    """Console script entry point referenced by setup.py for system bootstrapping."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting isolated Qdrant index validation routine...")
    qdrant_manager.bootstrap_collection()
    logger.info("Index validation phase complete.")

if __name__ == "__main__":
    main()
