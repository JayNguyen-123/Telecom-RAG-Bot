# Connection pooling logic (Psycopg2/SQLAlchemy)
import logging
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extensions import cursor as psycopg2_cursor
from psycopg2.pool import ThreadedConnectionPool

from config.settings import settings

# Initialize runtime logger
logger = logging.getLogger(__name__)

class PostgresConnectionManager:
    """
    Manages a thread-safe PostgreSQL connection pool for the RAG lifecycle.
    Implements a singleton pattern to ensure only one pool is instantiated.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostgresConnectionManager, cls).__new__(cls)
            cls._instance._pool = None
        return cls._instance

    def initialize_pool(self) -> None:
        """Initializes the database connection pool using validated settings configuration."""
        if self._pool is not None:
            return

        try:
            # Cast PostgresDsn validation object to a standard string
            dsn_str = str(settings.DB_CONNECTION_STRING)
            
            logger.info("Initializing PostgreSQL ThreadedConnectionPool...")
            self._pool = ThreadedConnectionPool(
                minconn=settings.DB_POOL_MIN_CONNECTIONS,
                maxconn=settings.DB_POOL_MAX_CONNECTIONS,
                dsn=dsn_str
            )
            logger.info("PostgreSQL connection pool initialized successfully.")
        except psycopg2.Error as e:
            logger.critical(f"Database bootstrap initialization failed: {e}")
            raise RuntimeError("Could not establish relational database infrastructure pool.") from e

    @contextmanager
    def get_connection(self) -> Generator[psycopg2_connection, None, None]:
        """
        Context manager that yields an active database connection from the pool.
        Automatically handles rollbacks on exception and releases the connection back to the pool.
        """
        if self._pool is None:
            self.initialize_pool()

        connection = self._pool.getconn()
        # Ensure the connection is set to commit transaction blocks explicitly
        connection.autocommit = False
        
        try:
            yield connection
            # If no exceptions were raised, commit the queries safely
            connection.commit()
        except Exception as e:
            logger.error(f"Database transaction failure. Rolling back database state. Error: {e}")
            connection.rollback()
            raise e
        finally:
            # Always return the connection handle back to the pool pool, even if a crash occurred
            self._pool.putconn(connection)

    @contextmanager
    def get_cursor(self) -> Generator[psycopg2_cursor, None, None]:
        """
        Context manager that directly yields an operational cursor block.
        Simplifies application level logic for standard data manipulation execution queries.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                yield cursor

    def close_all_connections(self) -> None:
        """Gracefully closes all open connections within the pool during application shutdown."""
        if self._pool is not None:
            logger.info("Closing all active PostgreSQL pool connections...")
            self._pool.closeall()
            self._pool = None
            logger.info("PostgreSQL connection pool terminated cleanly.")

# Instantiate a single, reusable connection manager instance
db_manager = PostgresConnectionManager()
