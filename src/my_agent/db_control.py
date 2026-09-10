"""Vector database control backed by Postgres + pgvector, with a custom embeddings table.

SQLAlchemy is used as the ORM (Object-Relational Mapping) to interact with the Postgres database,
while pgvector provides support for vector embeddings.
"""
import json
import os
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pgvector.sqlalchemy import Vector
# SQLAlchemy and pgvector imports for ORM and vector support.
from sqlalchemy import String, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# Must match the output dimension of the embedding model in use (e.g. Titan v2 default is 1024).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


def _sanitize_metadata(metadata: dict) -> dict:
    """Coerce metadata values (e.g. PosixPath) into JSON-serializable types.

    Args:
        metadata (dict): The raw document metadata.

    Returns:
        dict: A JSON-serializable copy of the metadata.

    """
    return json.loads(json.dumps(metadata, default=str))


def build_connection_string() -> str:
    """Build the Postgres connection string from environment variables.

    Returns:
        str: A psycopg-compatible SQLAlchemy connection string.

    """
    user = os.getenv("POSTGRES_USER", "rag_user")
    password = os.getenv("POSTGRES_PASSWORD", "rag_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "rag_db")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


class Base(DeclarativeBase):
    """Declarative base for the vector database ORM models."""


class EmbeddingRow(Base):
    """ORM model for the `embeddings` table.
    
    This table stores the embeddings of document chunks along with
    metadata and the model used to generate the embeddings. It is used by
    the `EmbeddingDBControl` class to manage storage and retrieval of embeddings.
    """

    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    # Attribute renamed to avoid clashing with SQLAlchemy's reserved `Base.metadata`.
    row_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    model_used: Mapped[str] = mapped_column(String, nullable=False)


@dataclass
class EmbeddingDBControl:
    """Manage embeddings storage and retrieval in a pgvector-backed `embeddings` table."""

    embeddings: Embeddings
    model_name: str
    connection: str = field(default_factory=build_connection_string)

    def __post_init__(self):
        self.engine = create_engine(self.connection)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store a list of documents in the `embeddings` table.

        Args:
            documents (list[Document]): The documents to embed and store.

        Returns:
            list[str]: The ids assigned to the stored rows.

        """
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        rows = [
            EmbeddingRow(
                chunk_content=doc.page_content,
                embedding=vector,
                row_metadata=_sanitize_metadata(doc.metadata),
                model_used=self.model_name,
            )
            for doc, vector in zip(documents, vectors)
        ]
        with Session(self.engine) as session:
            session.add_all(rows)
            session.commit()
            return [row.id for row in rows]

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for the k most similar rows to the query, by cosine distance.

        Args:
            query (str): The text to search for.
            k (int, optional): The number of results to return. Defaults to 4.

        Returns:
            list[Document]: The most similar rows, converted back to Documents.

        """
        query_vector = self.embeddings.embed_query(query)
        with Session(self.engine) as session:
            rows = (
                session.query(EmbeddingRow)
                .order_by(EmbeddingRow.embedding.cosine_distance(query_vector))
                .limit(k)
                .all()
            )
            return [
                Document(
                    page_content=row.chunk_content,
                    metadata={**row.row_metadata, "model_used": row.model_used},
                )
                for row in rows
            ]

    def status(self) -> dict:
        """Check the database connection and report basic health information.

        Returns:
            dict: `connected`, `vector_extension`, `row_count` and `sample_row`
                (all fields of one stored row, or None if the table is empty),
                plus an `error` message when `connected` is False.

        """
        try:
            with self.engine.connect() as conn:
                has_extension = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ).first() is not None
                row_count = conn.execute(func.count(EmbeddingRow.id).select()).scalar()
            with Session(self.engine) as session:
                sample = session.query(EmbeddingRow).first()
                sample_row = (
                    {
                        "id": sample.id,
                        "chunk_content": sample.chunk_content,
                        "embedding": sample.embedding,
                        "row_metadata": sample.row_metadata,
                        "model_used": sample.model_used,
                    }
                    if sample
                    else None
                )
            return {
                "connected": True,
                "vector_extension": has_extension,
                "row_count": row_count,
                "sample_row": sample_row,
            }
        except Exception as exc:  # noqa: BLE001 - surface any connection/db error to the caller
            return {"connected": False, "error": str(exc)}

    def delete(self, ids: list[str]) -> None:
        """Delete rows from the `embeddings` table by id.

        Args:
            ids (list[str]): The ids of the rows to delete.

        """
        with Session(self.engine) as session:
            session.query(EmbeddingRow).filter(EmbeddingRow.id.in_(ids)).delete(synchronize_session=False)
            session.commit()
