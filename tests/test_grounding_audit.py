"""Grounding-audit schema and orchestration tests."""

import pytest
from pydantic import ValidationError

from relay.nodes.schemas import GroundingAuditOutput


def test_grounded_audit_rejects_unsupported_claims() -> None:
    with pytest.raises(ValidationError):
        GroundingAuditOutput(
            assessment="grounded",
            unsupported_claims=["Unsupported claim."],
            rationale="Invalid combination.",
        )


def test_revision_audit_requires_an_issue() -> None:
    with pytest.raises(ValidationError):
        GroundingAuditOutput(
            assessment="needs_revision",
            unsupported_claims=[],
            rationale="Invalid combination.",
        )
