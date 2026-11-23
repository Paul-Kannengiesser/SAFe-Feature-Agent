"""Run log for the evaluation: one JSON file per dialogue in runs/, rewritten every turn (see RUNS.md)."""

import json
from pathlib import Path

from .agent import Agent, _key
from .config import AGENT_DIR, settings
from .schema import FIELD_LABELS, FIELD_ORDER

RUNS_DIR = AGENT_DIR / "runs"


def run_path(agent: Agent) -> Path:
    return RUNS_DIR / f"run_{agent.started:%Y-%m-%d_%H-%M-%S}.json"


def save(agent: Agent) -> Path:
    open_keys = {_key(f) for f in agent.findings}
    accepted_keys = {_key(f) for f in agent.accepted}
    resolved = [f for k, f in agent.seen.items() if k not in open_keys | accepted_keys]
    seconds = [t["seconds"] for t in agent.turns]

    data = {
        "run": run_path(agent).stem,
        "started": agent.started.isoformat(timespec="seconds"),
        "settings": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "nli_model": settings.nli_model,
            "data_source": settings.data_source,
            "features_file": str(settings.features_file),
            "epics_file": str(settings.epics_file),
        },
        "epic": {
            "candidates": [f"{e.key}: {e.name}" for e in agent.epic_candidates],
            "chosen": f"{agent.epic.key}: {agent.epic.name}" if agent.epic else None,
        },
        "possible_duplicates": [{"key": h.item.key, "name": h.item.name, "similarity": round(h.similarity, 2)}
                                for h in agent.overlaps],
        "turns": agent.turns,
        "bridged": agent.bridged,
        "result": {
            "complete": agent.is_done(),
            "complete_after_pm_messages": agent.completed_after,
            "ticket": {FIELD_LABELS[f]: agent.draft.field_text(f) for f in FIELD_ORDER},
            "ticket_with_provenance": agent.draft.model_dump(),
            "name_confirmed": agent.name_confirmed,
            "issues": {
                "open": agent.issue_dicts(agent.findings),
                "accepted": agent.issue_dicts(agent.accepted),
                "resolved": agent.issue_dicts(resolved),
            },
            "pm_messages": len(agent.turns),
            "seconds_total": round(sum(seconds), 1),
            "seconds_per_turn_avg": round(sum(seconds) / len(seconds), 1) if seconds else 0,
        },
    }
    RUNS_DIR.mkdir(exist_ok=True)
    path = run_path(agent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
