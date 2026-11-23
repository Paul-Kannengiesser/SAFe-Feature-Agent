"""Parsing of raw Jira issues (building block 1).

Benefit hypothesis and acceptance criteria are headed sections of the description.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from .config import settings

IssueType = Literal["Feature", "Epic"]

BENEFIT_TITLES = {"benefit hypothesis", "benefit hypotheses", "benefit"}
ACCEPTANCE_TITLES = {"acceptance criteria", "acceptance criterion", "akzeptanzkriterien"}

# synonyms seen in the export
TITLE_ALIASES = {
    "problem statement/ description": "description",
    "problem statement/description": "description",
    "problem statement": "description",
}

# bold-only lines ("*Acceptance Criteria.*") count as headings only for these titles
KNOWN_SECTION_TITLES = (
    {"description", "in scope", "out of scope", "dependencies", "external dependencies",
     "open points", "customer journey", "what will happen if not implemented?",
     "links/ relevant content"}
    | BENEFIT_TITLES | ACCEPTANCE_TITLES | set(TITLE_ALIASES)
)

_HEADING = re.compile(r"^\s*(?:h[1-6]\.\s*)+(.*)$")
_BOLD_LINE = re.compile(r"^\s*\*([^*]{3,60})\*\s*:?\s*$")
_PI = re.compile(r"PI\s*-?\s*0*(\d+)", re.IGNORECASE)


class BacklogItem(BaseModel):
    key: str
    issue_type: IssueType
    name: str
    status: str
    pi: int | None = None
    epic_key: str | None = None  # only set for features
    description: str = ""  # plain text, wiki markup removed
    sections: dict[str, str] = {}  # normalized section title -> plain text
    updated: datetime | None = None

    @property
    def benefit_hypothesis(self) -> str | None:
        return _first_section(self.sections, BENEFIT_TITLES)

    @property
    def acceptance_criteria(self) -> str | None:
        return _first_section(self.sections, ACCEPTANCE_TITLES)

    def header(self) -> str:
        meta = [self.issue_type, self.status]
        if self.pi is not None:
            meta.append(f"PI {self.pi}")
        if self.epic_key:
            meta.append(f"Epic {self.epic_key}")
        return f"{self.key} [{', '.join(meta)}] {self.name}"

    def context_text(self, max_chars: int = 1500) -> str:
        """Compact text representation used for embeddings and LLM context."""
        text = self.header()
        if self.description:
            text += "\n" + self.description
        return text[:max_chars]


def _first_section(sections: dict[str, str], titles: set[str]) -> str | None:
    for title, body in sections.items():
        if title in titles and body.strip():
            return body.strip()
    return None


def normalize_title(title: str) -> str:
    title = re.sub(r"[*_:{}]", "", title).lower()
    title = title.replace("(optional)", "")
    title = re.sub(r"\s+", " ", title).strip().rstrip(".").strip()
    return TITLE_ALIASES.get(title, title)


def _heading_title(line: str) -> str | None:
    """Return the normalized title if the line is a section heading."""
    if match := _HEADING.match(line):
        return normalize_title(match.group(1))
    if match := _BOLD_LINE.match(line):
        title = normalize_title(match.group(1))
        if title in KNOWN_SECTION_TITLES:
            return title
    return None


def wiki_to_text(markup: str) -> str:
    """Convert Jira wiki markup into readable plain text (best effort)."""
    lines = []
    for line in markup.replace("\r\n", "\n").split("\n"):
        heading = _HEADING.match(line)
        if heading:
            line = heading.group(1)
        line = re.sub(r"^\s*[*#-]+\s+", "- ", line)  # list items
        line = re.sub(r"\{(?:color|noformat|code|quote|panel)[^}]*\}", "", line)
        line = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", line)  # [text|link] -> text
        line = re.sub(r"(?<!\w)[*_+](\S(?:.*?\S)?)[*_+](?!\w)", r"\1", line)  # bold/italic
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_sections(markup: str | None) -> dict[str, str]:
    """Split a wiki-markup description at its headings; text before the first one goes under ""."""
    if not markup:
        return {}
    sections: dict[str, list[str]] = {}
    current = ""
    for line in markup.replace("\r\n", "\n").split("\n"):
        title = _heading_title(line)
        if title is not None:
            current = title
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    result = {title: wiki_to_text("\n".join(body)) for title, body in sections.items()}
    return {title: body for title, body in result.items() if body or title}


def parse_pi(raw: Any) -> int | None:
    value = raw.get("value") if isinstance(raw, dict) else raw
    if not isinstance(value, str):
        return None
    match = _PI.search(value)
    return int(match.group(1)) if match else None


def parse_issue(raw: dict[str, Any]) -> BacklogItem:
    """Convert one raw issue from a Jira search response into a BacklogItem."""
    f = raw["fields"]
    fid = settings.fields
    issue_type = f["issuetype"]["name"]
    if issue_type not in ("Feature", "Epic"):
        raise ValueError(f"{raw.get('key')}: unsupported issue type {issue_type!r}")
    description = f.get("description") or ""
    updated = f.get("updated")
    # Features use both PI fields in practice; epics only the epic PI field.
    if issue_type == "Feature":
        pi = parse_pi(f.get(fid.feature_pi)) or parse_pi(f.get(fid.epic_pi))
    else:
        pi = parse_pi(f.get(fid.epic_pi))
    return BacklogItem(
        key=raw["key"],
        issue_type=issue_type,
        name=(f.get("summary") or "").strip(),
        status=f["status"]["name"],
        pi=pi,
        epic_key=f.get(fid.epic_link) if issue_type == "Feature" else None,
        description=wiki_to_text(description),
        sections=split_sections(description),
        updated=datetime.strptime(updated, "%Y-%m-%dT%H:%M:%S.%f%z") if updated else None,
    )
