"""Add priority column to dataset_samples

Revision ID: a1b2c3d4e5f6
Revises: f9f91aa870ec
Create Date: 2026-06-28 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f9f91aa870ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'dataset_samples',
        sa.Column('priority', sa.Integer(), nullable=False, server_default='1')
    )


def downgrade() -> None:
    op.drop_column('dataset_samples', 'priority')
