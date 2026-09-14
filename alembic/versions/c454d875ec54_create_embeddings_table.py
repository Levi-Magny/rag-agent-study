"""create embeddings table

Revision ID: c454d875ec54
Revises: 
Create Date: 2026-09-10 21:08:29.767797

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from my_agent.config import get_settings

# revision identifiers, used by Alembic.
revision: str = 'c454d875ec54'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embeddings",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("chunk_content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(get_settings().embedding_dim), nullable=False),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column("model_used", sa.String(), nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("embeddings")
