"""Data sources (building block 1): live Jira or a snapshot export from `jira_rohdaten.py`.

Both take an `IssueFilter`; the live source sends it as JQL, the file source applies it directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import requests
from pydantic import BaseModel, Field

from .backlog import BacklogItem, IssueType, parse_issue
from .config import settings


class IssueFilter(BaseModel):
    """Structured query against the backlog; all set criteria are AND-combined."""

    issue_type: IssueType | None = None
    keys: list[str] = Field(default_factory=list)
    epic_key: str | None = Field(None, description="Only features linked to this epic")
    statuses: list[str] = Field(default_factory=list)
    pi: int | None = Field(None, description="Program Increment number, e.g. 2")
    text: str | None = Field(None, description="Words that must occur in name or description")

    def to_jql(self, project: str = settings.jira_project) -> str:
        fid = settings.fields
        clauses = [f"project = {project}"]
        if self.issue_type:
            clauses.append(f'issuetype = "{self.issue_type}"')
        else:
            clauses.append('issuetype in ("Feature", "Epic")')
        if self.keys:
            clauses.append(f"key in ({', '.join(self.keys)})")
        if self.epic_key:
            clauses.append(f'"Epic Link" = {self.epic_key}')
        if self.statuses:
            clauses.append("status in (" + ", ".join(f'"{s}"' for s in self.statuses) + ")")
        if self.pi is not None:
            # Feature and epic PI live in different custom fields with different spelling
            feature_cf = fid.feature_pi.removeprefix("customfield_")
            epic_cf = fid.epic_pi.removeprefix("customfield_")
            clauses.append(
                f'(cf[{feature_cf}] = "PI {self.pi}" OR cf[{epic_cf}] = "PI-{self.pi:02d}")'
            )
        if self.text:
            clauses.append(f'text ~ "{_escape_jql(self.text)}"')
        return " AND ".join(clauses) + " ORDER BY key ASC"

    def matches(self, item: BacklogItem) -> bool:
        if self.issue_type and item.issue_type != self.issue_type:
            return False
        if self.keys and item.key not in self.keys:
            return False
        if self.epic_key and item.epic_key != self.epic_key:
            return False
        if self.statuses and item.status.lower() not in {s.lower() for s in self.statuses}:
            return False
        if self.pi is not None and item.pi != self.pi:
            return False
        if self.text:
            haystack = f"{item.name}\n{item.description}".lower()
            # approximates JQL "text ~": every word must occur
            if not all(word in haystack for word in _words(self.text)):
                return False
        return True


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\w.-]+", text.lower()) if len(w) > 1]


def _escape_jql(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


class BacklogSource(Protocol):
    def search(self, query: IssueFilter) -> list[BacklogItem]: ...

    def all_items(self) -> list[BacklogItem]: ...


def _load_export(path: Path) -> list[dict]:
    """Load a Jira search response and any additional page files next to it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Export not found: {path}")
    files = [path, *sorted(path.parent.glob(f"{path.stem}_part*{path.suffix}"))]
    issues: list[dict] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        issues.extend(data["issues"] if isinstance(data, dict) and "issues" in data else data)
    return issues


class FileSource:
    def __init__(self, features_file: Path | None = None, epics_file: Path | None = None):
        raw = _load_export(features_file or settings.features_file)
        raw += _load_export(epics_file or settings.epics_file)
        items = {item.key: item for item in map(parse_issue, raw)}
        self._items = sorted(items.values(), key=_key_order)

    def all_items(self) -> list[BacklogItem]:
        return list(self._items)

    def search(self, query: IssueFilter) -> list[BacklogItem]:
        return [item for item in self._items if query.matches(item)]


def _key_order(item: BacklogItem) -> tuple[str, int]:
    project, _, number = item.key.partition("-")
    return project, int(number) if number.isdigit() else 0


class JiraSource:
    PAGE_SIZE = 100

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or settings.jira_url).rstrip("/")
        token = token or settings.jira_token
        if not token:
            raise ValueError("Jira token missing (set FSEA_JIRA_TOKEN)")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def all_items(self) -> list[BacklogItem]:
        return self.search(IssueFilter())

    def search(self, query: IssueFilter) -> list[BacklogItem]:
        jql = query.to_jql()
        items: list[BacklogItem] = []
        start_at = 0
        while True:
            response = self.session.get(
                f"{self.base_url}/rest/api/2/search",
                params={"jql": jql, "fields": "*all", "startAt": start_at,
                        "maxResults": self.PAGE_SIZE},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            issues = data.get("issues", [])
            items.extend(parse_issue(issue) for issue in issues)
            start_at += len(issues)
            if not issues or start_at >= data.get("total", 0):
                return items


def get_source() -> BacklogSource:
    if settings.data_source == "jira":
        return JiraSource()
    return FileSource()
