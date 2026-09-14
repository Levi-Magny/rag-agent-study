"""add embedding source identity

Revision ID: 99d6bc53b244
Revises: c454d875ec54
Create Date: 2026-09-10

"""
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "99d6bc53b244"
down_revision: Union[str, Sequence[str], None] = "c454d875ec54"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _source_id(source_path: str) -> str:
    normalized_path = Path(source_path).expanduser().resolve(strict=False).as_posix()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()


def upgrade() -> None:
    """Add an indexed, unique chunk sequence for every source document."""
    op.add_column("embeddings", sa.Column("source_id", sa.String(), nullable=True))
    op.add_column("embeddings", sa.Column("chunk_index", sa.Integer(), nullable=True))

    connection = op.get_bind()
    rows = connection.execute(
        sa.text("SELECT id, metadata FROM embeddings ORDER BY id")
    ).mappings()
    source_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        metadata = row["metadata"] or {}
        source_path = metadata.get("dl_meta", {}).get("origin", {}).get("filename")
        source_id = _source_id(source_path) if source_path else f"legacy:{row['id']}"
        chunk_index = source_counts[source_id]
        source_counts[source_id] += 1
        connection.execute(
            sa.text(
                "UPDATE embeddings SET source_id = :source_id, chunk_index = :chunk_index "
                "WHERE id = :id"
            ),
            {"source_id": source_id, "chunk_index": chunk_index, "id": row["id"]},
        )

    op.alter_column("embeddings", "source_id", nullable=False)
    op.alter_column("embeddings", "chunk_index", nullable=False)
    op.create_index("ix_embeddings_source_id", "embeddings", ["source_id"])
    op.create_unique_constraint(
        "uq_embeddings_source_chunk", "embeddings", ["source_id", "chunk_index"]
    )


def downgrade() -> None:
    """Remove document source identity columns."""
    op.drop_constraint("uq_embeddings_source_chunk", "embeddings", type_="unique")
    op.drop_index("ix_embeddings_source_id", table_name="embeddings")
    op.drop_column("embeddings", "chunk_index")
    op.drop_column("embeddings", "source_id")