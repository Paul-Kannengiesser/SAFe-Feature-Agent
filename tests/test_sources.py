from pathlib import Path

import pytest

from feature_agent.backlog import parse_pi, split_sections, wiki_to_text
from feature_agent.sources import FileSource, IssueFilter

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def source() -> FileSource:
    return FileSource(FIXTURES / "features.json", FIXTURES / "epics.json")


def test_split_sections_handles_duplicated_heading_prefix():
    markup = "h2. h2. Description\r\n\r\nSome text.\r\nh2. Benefit Hypothesis\r\n\r\nValue.\r\n"
    assert split_sections(markup) == {"description": "Some text.", "benefit hypothesis": "Value."}


def test_split_sections_bold_headings_and_aliases():
    markup = (
        "h2. Problem statement/ Description\nText.\n"
        "*Acceptance Criteria.*\n * AC one\n"
        "*Important*\nnot a heading\n"
        "h3. Out of Scope (optional)\n * nothing\n"
    )
    assert split_sections(markup) == {
        "description": "Text.",
        "acceptance criteria": "- AC one\nImportant\nnot a heading",
        "out of scope": "- nothing",
    }


def test_split_sections_keeps_preamble_without_heading():
    assert split_sections("Just a sentence.") == {"": "Just a sentence."}


def test_wiki_to_text_converts_lists_and_links():
    text = wiki_to_text(" * first\r\n * [Docs|https://example.org] are *important*")
    assert text == "- first\n- Docs are important"


@pytest.mark.parametrize("raw, expected", [
    ({"value": "PI 2"}, 2), ({"value": "PI-02"}, 2), ("PI 12", 12), (None, None), ({"value": "n/a"}, None),
])
def test_parse_pi(raw, expected):
    assert parse_pi(raw) == expected


def test_file_source_loads_and_parses(source):
    items = {i.key: i for i in source.all_items()}
    assert len(items) == 10
    pdf = items["DEMO-10"]
    assert pdf.issue_type == "Feature"
    assert pdf.epic_key == "DEMO-1"
    assert pdf.pi == 2
    assert pdf.benefit_hypothesis.startswith("Exporting reports as PDF")
    assert pdf.acceptance_criteria.splitlines()[0] == "- When an inspection is finished, the export button is enabled"
    assert items["DEMO-11"].acceptance_criteria is None
    assert items["DEMO-1"].pi == 2 and items["DEMO-1"].epic_key is None


def test_filter_by_epic_and_status(source):
    keys = [i.key for i in source.search(IssueFilter(epic_key="DEMO-1", statuses=["open"]))]
    assert keys == ["DEMO-10", "DEMO-11"]


def test_filter_by_pi_includes_epics_and_features(source):
    keys = [i.key for i in source.search(IssueFilter(pi=3))]
    assert keys == ["DEMO-2", "DEMO-20", "DEMO-21"]


def test_filter_by_text(source):
    keys = [i.key for i in source.search(IssueFilter(text="password reset"))]
    assert keys == ["DEMO-20"]


def test_to_jql():
    jql = IssueFilter(issue_type="Feature", epic_key="DEMO-1", pi=2, text='say "hi"').to_jql(project="DEMO")
    assert jql == (
        'project = DEMO AND issuetype = "Feature" AND "Epic Link" = DEMO-1 AND '
        '(cf[16355] = "PI 2" OR cf[16351] = "PI-02") AND text ~ "say \\"hi\\"" ORDER BY key ASC'
    )


def test_real_samples_parse_if_present():
    jira_dir = Path(__file__).parent.parent / "jira"
    if not (jira_dir / "feature.json").exists():
        pytest.skip("no local Jira samples")
    items = {i.key: i for i in FileSource(jira_dir / "feature.json", jira_dir / "epic.json").all_items()}
    feature = next(i for i in items.values() if i.issue_type == "Feature")
    assert feature.benefit_hypothesis
    assert feature.epic_key in items
