# Structured runtime application logging
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

from config.settings import settings

class JSONStructuredFormatter(logging.Formatter):
    """
    Custom logging formatter that serializes application trace properties
    into machine-readable, single-line JSON metric strings.
    """
    def __init__(self, environment: str, project_name: str):
        super().__init__()
        self.environment = environment
        self.project_name = project_name

    def format(self, record: logging.LogRecord) -> str:
        # Standardize modern timestamp structure down to UTC Isoformat
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        # Build the structured dictionary payload contract
        log_payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "project": self.project_name,
            "environment": self.environment,
            "logger_name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread
        }

        # Automatically extract stack trace metadata blocks if an exception occurs
        if record.exc_info:
            log_payload["exception_details"] = self.formatException(record.exc_info)

        # Merge inline contextual custom fields provided via the logging 'extra' parameter
        # Example: logger.info("message", extra={"session_id": "xyz"})
        if hasattr(record, "__dict__"):
            standard_record_keys = {
                'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'msg', 'name', 'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'thread', 'threadName'
            }
            extra_payload = {
                k: v for k, v in record.__dict__.items()
                if k not in standard_record_keys
            }
            if extra_payload:
                log_payload["context_metrics"] = extra_payload

        return json.dumps(log_payload)


def configure_production_logging() -> None:
    """
    Overrides the runtime environment logging configuration topology.
    Hooks into standard output pipes and applies structural JSON boundaries.
    """
    # Fetch log target constraints from our centralized Pydantic settings singleton
    log_level_string = settings.LOG_LEVEL
    numeric_level = getattr(logging, log_level_string.upper(), logging.INFO)

    # Initialize our specialized structural encoder
    json_formatter = JSONStructuredFormatter(
        environment=settings.ENVIRONMENT,
        project_name=settings.PROJECT_NAME
    )

    # Stream structured logs straight out to standard stream outputs
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(json_formatter)

    # Configure the global logging root configuration level boundary parameters
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Wipe out any default plain-text handlers inherited during runtime bootstrap steps
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    root_logger.addHandler(stdout_handler)

    # Minimize internal chatter levels from third-party vendor library frameworks
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)

    logging.info(f"System logging interface successfully synchronized to JSON format mode. Active Level: {log_level_string}")

