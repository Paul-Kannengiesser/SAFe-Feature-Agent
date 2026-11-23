# RAG-basierter Dialogagent für SAFe-Features (Prototyp)

Lokaler Dialogagent, der einen Product Manager im Dialog zu einem SAFe-Feature mit Name,
Benefit Hypothesis und Acceptance Criteria führt. Bestehende Epics und Features aus Jira
dienen als Kontext, die Inhalte werden auf Eindeutigkeit, Konsistenz, Nachvollziehbarkeit
und Testbarkeit geprüft. Alles läuft lokal, es verlassen keine Daten das Gerät.

## Start

Voraussetzungen: Python 3.12 oder neuer und [Ollama](https://ollama.com).

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
ollama pull qwen3:30b-a3b-q8_0 && ollama pull qwen3-embedding:0.6b
.venv/bin/streamlit run app.py
```

Das NLI-Modell (DeBERTa-v3) wird beim ersten Start von Hugging Face geladen.

## Datenquelle

Standardmäßig liest der Agent einen Jira-Export aus `jira/features.json` und
`jira/epics.json`, andere Pfade über `FSEA_FEATURES_FILE` und `FSEA_EPICS_FILE`.
Einen Export erzeugt `jira_rohdaten.py`. Zum Ausprobieren ohne echte Daten:

```bash
FSEA_FEATURES_FILE=tests/fixtures/features.json FSEA_EPICS_FILE=tests/fixtures/epics.json \
  .venv/bin/streamlit run app.py
```

Für den Live-Zugriff auf Jira: `FSEA_DATA_SOURCE=jira`, `FSEA_JIRA_URL`,
`FSEA_JIRA_PROJECT` und `FSEA_JIRA_TOKEN`.

## Aufbau

| Datei | Aufgabe |
|---|---|
| `feature_agent/sources.py`, `backlog.py` | Datenquelle (Jira live oder Export) |
| `feature_agent/retrieval.py` | Retrieval (strukturiert und semantisch) |
| `feature_agent/schema.py` | Zielschema mit Herkunftsangaben |
| `feature_agent/agent.py` | Dialogsteuerung |
| `feature_agent/quality.py` | Qualitätsprüfung (Regeln, NLI, LLM) |
| `feature_agent/export.py` | Dialogprotokolle in `runs/`, Format in [RUNS.md](RUNS.md) |
| `app.py` | Weboberfläche |

## Tests

```bash
.venv/bin/pytest
```

Die Retrieval-Tests brauchen einen laufenden Ollama-Server und werden sonst übersprungen.
