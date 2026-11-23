"""Central configuration. Values can be overridden via environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class JiraFieldIds:
    """Custom field IDs of the Jira instance (Jira Data Center)."""

    epic_link: str = "customfield_10256"  # Feature -> Epic key, e.g. "DEMO-1"
    epic_name: str = "customfield_10257"  # Epic short name
    feature_pi: str = "customfield_16355"  # Feature PI, e.g. "PI 2"
    epic_pi: str = "customfield_16351"  # Epic PI, e.g. "PI-02"


@dataclass(frozen=True)
class Settings:
    # Data source: "file" (snapshot export) or "jira" (live REST API)
    data_source: str = os.environ.get("FSEA_DATA_SOURCE", "file")

    # File source; further pages (features_part2.json, ...) are picked up automatically
    features_file: Path = Path(
        os.environ.get("FSEA_FEATURES_FILE", AGENT_DIR / "jira" / "features.json")
    )
    epics_file: Path = Path(os.environ.get("FSEA_EPICS_FILE", AGENT_DIR / "jira" / "epics.json"))

    # Live source
    jira_url: str = os.environ.get("FSEA_JIRA_URL", "https://jira.example.com")
    jira_project: str = os.environ.get("FSEA_JIRA_PROJECT", "DEMO")
    jira_token: str | None = os.environ.get("FSEA_JIRA_TOKEN")

    # Models (all local via Ollama / Hugging Face)
    llm_model: str = os.environ.get("FSEA_LLM_MODEL", "qwen3:30b-a3b-q8_0")
    embedding_model: str = os.environ.get("FSEA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    nli_model: str = os.environ.get(
        "FSEA_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    )

    chroma_dir: Path = Path(os.environ.get("FSEA_CHROMA_DIR", AGENT_DIR / ".chroma"))

    fields: JiraFieldIds = field(default_factory=JiraFieldIds)


settings = Settings()
