"""Integration tests for the semantic path; require a running Ollama server."""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
import ollama
import pytest

from feature_agent.retrieval import Retriever
from feature_agent.schema import FeatureDraft, TracedText
from feature_agent.sources import FileSource, IssueFilter

FIXTURES = Path(__file__).parent / "fixtures"


def _ollama_available() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    if not _ollama_available():
        pytest.skip("Ollama not running")
    source = FileSource(FIXTURES / "features.json", FIXTURES / "epics.json")
    return Retriever(source, client=chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False)))


def test_semantic_finds_related_feature(retriever):
    hits = retriever.similar("Users who forgot their login credentials can recover access themselves", k=3)
    assert hits[0].item.key == "DEMO-20"
    assert all(h.item.issue_type == "Feature" for h in hits)


def test_semantic_can_exclude_epic(retriever):
    hits = retriever.similar("sign the inspection report digitally", k=3, exclude_epic="DEMO-1")
    assert all(h.item.epic_key != "DEMO-1" for h in hits)


def test_index_is_reused_when_backlog_unchanged(retriever):
    before = retriever.collection.metadata["fingerprint"]
    assert retriever._ensure_index().metadata["fingerprint"] == before


def test_structured_path(retriever):
    assert [i.key for i in retriever.structured(IssueFilter(epic_key="DEMO-2"))] == ["DEMO-20", "DEMO-21"]


def test_feature_draft_missing_fields():
    draft = FeatureDraft(name=TracedText(value="X | Y"))
    assert draft.missing_fields() == ["benefit_hypothesis", "acceptance_criteria"]
    draft.acceptance_criteria = [TracedText(value=" ")]
    assert not draft.is_filled("acceptance_criteria")
    draft.benefit_hypothesis = TracedText(value="b")
    draft.acceptance_criteria = [TracedText(value="a1"), TracedText(value="a2")]
    assert draft.is_complete()
    assert draft.field_text("acceptance_criteria") == "- a1\n- a2"
