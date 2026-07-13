# Gemini native context caching lifecycle management
import logging
from typing import Optional, List
from google import genai
from google.genai import types
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)

class GeminiContextCacheManager:
    """
    Manages server-side static context caching structures for the Google GenAI SDK.
    Optimizes network costs and token pricing metrics for large data blocks.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiContextCacheManager, cls).__new__(cls)
            # Initialize the raw native Google GenAI SDK client core
            cls._instance.client = genai.Client()
        return cls._instance

    def _generate_cache_key(self, plan_tier: str) -> str:
        """Constructs a deterministic system naming identifier for target context keys."""
        return f"telecom_manual_{plan_tier.lower()}"

    def get_active_cache_handle(self, plan_tier: str) -> Optional[str]:
        """
        Scans Google Cloud server allocations to see if a cache container 
        already exists for the customer's plan tier.
        """
        target_display_name = self._generate_cache_key(plan_tier)
        try:
            # Page through active cached items on the cloud project instance
            for cache in self.client.caches.list():
                if cache.display_name == target_display_name:
                    logger.info(f"Located existing server context cache reference: {cache.name}")
                    return cache.name
            return None
        except Exception as e:
            logger.error(f"Failed to scan cloud context cache directory: {e}")
            return None

    def provision_context_cache(self, plan_tier: str, documents: List[Document]) -> Optional[str]:
        """
        Combines a list of documents and registers a server-side context cache container.
        Returns the resource name handle string used to intercept downstream calls.
        """
        if not documents:
            return None

        # 1. Check if a valid matching cache is already warm on Google's clusters
        existing_handle = self.get_active_cache_handle(plan_tier)
        if existing_handle:
            return existing_handle

        # 2. Flatten document arrays into a single, unified text corpus block
        combined_contents = "\n\n=== REFEFERENCE DOCUMENT MANUAL ===\n\n".join(
            [d.page_content for d in documents]
        )

        display_name = self._generate_cache_key(plan_tier)
        ttl_string = f"{settings.CONTEXT_CACHE_TTL_SECONDS}s"

        logger.warning(f"No existing context cache found for tier: {plan_tier}. Compiling fresh server instance...")
        
        try:
            # 3. Create the remote cache structure on Google GenAI infrastructure
            cache = self.client.caches.create(
                model=settings.LLM_MODEL_NAME,
                config=types.CreateCachedContentConfig(
                    contents=[combined_contents],
                    display_name=display_name,
                    ttl=ttl_string, 
                )
            )
            logger.info(f"Successfully generated remote context container. Name Handle: {cache.name}")
            return cache.name
            
        except Exception as e:
            logger.error(f"Abrupt crash provisioning server-side context token stream cache: {e}")
            # Fall back gracefully to non-cached prompt engineering execution
            return None

    def purge_expired_caches(self) -> None:
        """
        Utility maintenance cleanup function designed to run via internal 
        cron frameworks to clean obsolete or expired tokens explicitly.
        """
        logger.info("Executing scheduled cleaning task on context cache allocations...")
        try:
            for cache in self.client.caches.list():
                # Check for expired structures or manually purge based on company requirements
                logger.debug(f"Evaluating cache instance for removal safety: {cache.name}")
        except Exception as e:
            logger.error(f"Failed to cleanly purge storage allocations: {e}")

# Instantiate global singleton handle 
cache_manager = GeminiContextCacheManager()
