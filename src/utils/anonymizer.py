# Customer PII scrubbing tools (Regex/Presidio)
import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class TelecomPIIAnonymizer:
    """
    High-performance text sanitation utility to scrub customer PII before LLM dispatch.
    Uses pre-compiled regex matching patterns for optimal string processing latency.
    """
    def __init__(self):
        # Pre-compile regex expressions to optimize throughput inside the hot execution loop
        self.patterns = {
            "PHONE_NUMBER": re.compile(
                r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            "EMAIL_ADDRESS": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "CREDIT_CARD": re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            "IPV4_ADDRESS": re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-5]|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4][0-5]|[01]?\d\d?)\b'
            ),
            "MAC_ADDRESS": re.compile(
                r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'
            )
        }

    def anonymize_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Scrubs identifiable customer records from an incoming text string.
        Replaces targets with structural classification tokens and returns a mapping
        dictionary to allow for reversal (de-anonymization) before displaying back to the user.
        """
        if not text:
            return "", {}

        sanitized_text = text
        token_vault: Dict[str, str] = {}
        token_counter = 1

        # Iterate across regex definitions to neutralize matching items
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(sanitized_text)
            for match in set(matches):  # Deduplicate to avoid redundant swaps
                # Generate a unique structural entity token tag
                token_placeholder = f"[{pii_type}_{token_counter}]"

                # Vault the raw string entry for local downstream hydration
                token_vault[token_placeholder] = match

                # Replace occurrences within the active string instance
                sanitized_text = sanitized_text.replace(match, token_placeholder)
                token_counter += 1

        return sanitized_text, token_vault

    def deanonymize_text(self, anonymized_text: str, token_vault: Dict[str, str]) -> str:
        """
        Restores masked PII tokens back into the raw textual string framework.
        Used to re-populate real entity metrics on the secure client interface layer.
        """
        if not anonymized_text or not token_vault:
            return anonymized_text

        hydrated_text = anonymized_text
        # Iterate inverse loop tracking token boundaries
        for token_placeholder, raw_value in token_vault.items():
            hydrated_text = hydrated_text.replace(token_placeholder, raw_value)

        return hydrated_text

# Instantiate a global, reusable anonymization singleton
pii_anonymizer = TelecomPIIAnonymizer()

