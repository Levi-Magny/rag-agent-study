"""Vector database control backed by Postgres + pgvector, with a custom embeddings table.

SQLAlchemy is used as the ORM (Object-Relational Mapping) to interact with the Postgres database,
while pgvector provides support for vector embeddings.
"""
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pgvector.sqlalchemy import Vector
# SQLAlchemy and pgvector imports for ORM and vector support.
from sqlalchemy import Integer, String, Text, UniqueConstraint, create_engine, func, or_, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from my_agent.config import get_settings

# Must match the output dimension of the embedding model in use (e.g. Titan v2 default is 1024).
EMBEDDING_DIM = get_settings().embedding_dim


def _sanitize_metadata(metadata: dict) -> dict:
    """Coerce metadata values (e.g. PosixPath) into JSON-serializable types.

    Args:
        metadata (dict): The raw document metadata.

    Returns:
        dict: A JSON-serializable copy of the metadata.

    """
    return json.loads(json.dumps(metadata, default=str))


def source_id_for_path(source_path: str | Path) -> str:
    """Return a stable identifier for all chunks originating from a file."""
    normalized_path = Path(source_path).expanduser().resolve(strict=False).as_posix()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()


def _normalize_sections(sections: str | list[str]) -> list[str]:
    """Normalize one or more requested headings and reject empty values."""
    normalized_sections = [sections] if isinstance(sections, str) else sections
    if not normalized_sections or any(not section.strip() for section in normalized_sections):
        raise ValueError("sections must contain at least one non-empty heading")
    return normalized_sections


def build_connection_string() -> str:
    """Build the Postgres connection string from the centralized settings.

    Returns:
        str: A psycopg-compatible SQLAlchemy connection string.

    """
    return get_settings().postgres_connection_string


class Base(DeclarativeBase):
    """Declarative base for the vector database ORM models."""


class EmbeddingRow(Base):
    """Object-Relational Mapping (ORM) model for the `embeddings` table.
    
    This table stores the embeddings of document chunks along with
    metadata and the model used to generate the embeddings. It is used by
    the `EmbeddingDBControl` class to manage storage and retrieval of embeddings.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("source_id", "chunk_index", name="uq_embeddings_source_chunk"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    # Attribute renamed to avoid clashing with SQLAlchemy's reserved `Base.metadata`.
    row_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    model_used: Mapped[str] = mapped_column(String, nullable=False)


def _section_filter(sections: list[str]):
    """Build an OR predicate for normalized and original Docling headings."""
    filters = []
    for section in sections:
        filters.extend(
            [
                EmbeddingRow.row_metadata["section"].contains([section]),
                EmbeddingRow.row_metadata["dl_meta"]["headings"].contains([section]),
            ]
        )
    return or_(*filters)


@dataclass
class EmbeddingDBControl:
    """Manage embeddings storage and retrieval in a pgvector-backed `embeddings` table."""

    embeddings: Embeddings
    model_name: str
    connection: str = field(default_factory=build_connection_string)

    def __post_init__(self):
        # Schema (extension + tables) is owned by Alembic migrations; run
        # `alembic upgrade head` before using this class against a fresh database.
        self.engine = create_engine(self.connection)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store a standalone batch of documents in the `embeddings` table.

        Args:
            documents (list[Document]): The documents to embed and store.

        Returns:
            list[str]: The ids assigned to the stored rows.

        """
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if len(vectors) != len(documents):
            raise ValueError("The embedding provider returned an unexpected number of vectors")
        batch_source_id = f"manual:{uuid.uuid4()}"
        rows = [
            EmbeddingRow(
                source_id=batch_source_id,
                chunk_index=chunk_index,
                chunk_content=doc.page_content,
                embedding=vector,
                row_metadata=_sanitize_metadata(doc.metadata),
                model_used=self.model_name,
            )
            for chunk_index, (doc, vector) in enumerate(zip(documents, vectors))
        ]
        with Session(self.engine) as session:
            session.add_all(rows)
            session.commit()
            return [row.id for row in rows]

    def replace_document(self, source_path: str | Path, documents: list[Document]) -> list[str]:
        """Atomically replace every stored chunk associated with one source file.

        Embeddings are generated before the database transaction so an embedding
        failure leaves the previously indexed version untouched.

        Args:
            source_path: Path of the source document being reindexed.
            documents: Final chunks to embed and persist for the source.

        Returns:
            list[str]: The IDs assigned to the replacement rows.

        """
        # Ensure the source path is absolute and normalized before proceeding.
        source_path = Path(source_path).expanduser().resolve(strict=False)
        source_id = source_id_for_path(source_path)
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if len(vectors) != len(documents):
            raise ValueError("The embedding provider returned an unexpected number of vectors")

        rows = []
        # Construct the metadata for each chunk, including source ID, source path, and chunk index.
        for chunk_index, (document, vector) in enumerate(zip(documents, vectors)):
            metadata = {
                **document.metadata,
                "_source_id": source_id,
                "_source_path": source_path.as_posix(),
                "_chunk_index": chunk_index,
            }
            rows.append(
                EmbeddingRow(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    chunk_content=document.page_content,
                    embedding=vector,
                    row_metadata=_sanitize_metadata(metadata),
                    model_used=self.model_name,
                )
            )
        # Delete existing rows for the source and insert the new ones in a transaction.
        with Session(self.engine) as session:
            with session.begin():
                session.query(EmbeddingRow).filter(EmbeddingRow.source_id == source_id).delete(
                    synchronize_session=False
                )
                session.add_all(rows)
                session.flush()
                row_ids = [row.id for row in rows]
        return row_ids

    def delete_all_documents(self) -> int:
        """Delete every stored embedding and return the number of deleted rows.

        This operation is intentionally separate from document replacement. Use
        it only for an explicit full reindex, as it removes every source.

        Returns:
            int: The number of deleted embedding rows.

        """
        with Session(self.engine) as session:
            deleted_count = session.query(EmbeddingRow).delete(synchronize_session=False)
            session.commit()
        return deleted_count

    def get_chunks_by_sections(self, sections: str | list[str]) -> list[Document]:
        """Return chunks containing any requested heading in their metadata.

        Both the normalized ``section`` metadata and Docling's original
        ``dl_meta.headings`` are checked to support newly indexed and legacy
        chunks. Multiple headings use OR semantics.

        Args:
            sections: One heading or a list of headings to match exactly.

        Returns:
            list[Document]: Matching chunks ordered by source and chunk order.

        Raises:
            ValueError: If no non-empty heading is provided.

        """
        normalized_sections = _normalize_sections(sections)
        with Session(self.engine) as session:
            rows = (
                session.query(EmbeddingRow)
                .filter(_section_filter(normalized_sections))
                .order_by(EmbeddingRow.source_id, EmbeddingRow.chunk_index)
                .all()
            )
            return [
                Document(
                    page_content=row.chunk_content,
                    metadata={**row.row_metadata, "model_used": row.model_used},
                )
                for row in rows
            ]

    def similarity_search(self, query: str, k: int = 7) -> list[Document]:
        """Search for the k most similar rows to the query, by cosine distance.

        Args:
            query (str): The text to search for.
            k (int, optional): The number of results to return. Defaults to 7.

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

    def get_chunks_by_source(self, source_path: str | Path) -> list[Document]:
        """Retrieve document chunks from the database by their source file path.

        Args:
            source_path (str | Path): The path to the source document.

        Returns:
            list[Document]: A list of Document objects representing the chunks of the source document.

        """
        source_id = source_id_for_path(source_path)

        with Session(self.engine) as session:
            rows = (
                session.query(EmbeddingRow)
                .filter(EmbeddingRow.source_id == source_id)
                .order_by(EmbeddingRow.chunk_index)
                .all()
            )

        return [
            Document(
                page_content=row.chunk_content,
                metadata={
                    **row.row_metadata,
                    "id": row.id,
                    "model_used": row.model_used,
                },
            )
            for row in rows
        ]
