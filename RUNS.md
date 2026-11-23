# Dialogprotokolle (`runs/`)

Jeder Dialog in der Oberfläche wird als **eine JSON-Datei** in `runs/` gespeichert,
z. B. `run_2026-11-16_10-30-00.json` (Startzeitpunkt des Dialogs). Die Datei wird nach
**jeder** Nachricht des PM neu geschrieben, ein abgebrochener Dialog ist also auch vollständig
bis zur letzten Antwort enthalten. „New feature“ in der Seitenleiste startet eine neue Datei.

> `runs/` ist in der `.gitignore`: Die Dateien enthalten echte Backlog-Inhalte.

## Aufbau

```text
run                    Name der Datei / ID des Dialogs
started                Startzeitpunkt
settings               verwendete Modelle und Datenquelle (für die Reproduzierbarkeit)
epic
  candidates           die drei vorgeschlagenen Epics (semantische Suche)
  chosen               das gewählte Epic, oder null bei "none"
possible_duplicates    bestehende Features mit Ähnlichkeit >= 0,7 zur Idee (Key, Name, Ähnlichkeit)
turns[]                eine Zeile pro Nachricht des PM (siehe unten)
bridged[]              ein Eintrag pro manueller Überbrückung durch den PM (siehe unten)
result                 Endstand (siehe unten)
```

### `turns[]`: eine Runde

| Feld | Bedeutung |
|---|---|
| `time` | Zeitpunkt der Antwort |
| `pm` | Nachricht des PM |
| `agent` | Antwort des Agenten |
| `seconds` | Antwortzeit des Agenten in Sekunden |
| `phase` | `epic` (Epic-Auswahl) oder `fields` (Erhebung der Felder) |
| `target` | Feld, nach dem der Agent **als Nächstes** fragt (`name`, `benefit_hypothesis`, `acceptance_criteria`), `null` = alles fertig |
| `intent` | vom LLM erkannte Absicht des PM: `agrees` (stimmt dem besprochenen Feld zu, wird nur für die Bestätigung des Namens genutzt); `null` während der Epic-Auswahl |
| `changes` | was sich in dieser Runde geändert hat (Feldinhalte, Name bestätigt) |
| `llm_review_skipped` | `true`, wenn die LLM-Prüfung (Eindeutigkeit, Testbarkeit) in dieser Runde übersprungen wurde, weil das LLM nicht für jeden zu prüfenden Text genau eine Bewertung lieferte. Die Texte werden in der nächsten Runde erneut geprüft. |
| `llm_review_skipped_fields` | betroffene Felder (`benefit_hypothesis`, `acceptance_criteria`), sonst leer |
| `ticket` | Stand der drei Felder nach der Runde |
| `open_issues` | zu diesem Zeitpunkt offene Qualitätsprobleme |
| `accepted_issues` | bis dahin vom PM manuell überbrückte Probleme |

### `result`: Endstand

| Feld | Bedeutung |
|---|---|
| `complete` | `true`, wenn alle Felder gefüllt sind und der Agent nichts mehr fragt (offene Probleme, zu denen schon zweimal nachgefragt wurde, können dann noch bestehen) |
| `complete_after_pm_messages` | nach wie vielen Nachrichten des PM (inkl. Idee und Epic-Wahl) das Feature zum ersten Mal vollständig war; `null`, wenn nie |
| `ticket` | die drei Felder als Text, wie in der Oberfläche |
| `ticket_with_provenance` | dieselben Felder inkl. Herkunft: `user_turns` = Nummern der PM-Nachrichten (0 = erste Nachricht), `backlog_keys` = genutzte Jira-Issues |
| `name_confirmed` | hat der PM den vom Agenten vorgeschlagenen Namen bestätigt oder geändert? |
| `issues.open` | am Ende noch offene Probleme |
| `issues.accepted` | Probleme, die der PM mit „Keep as is“ manuell überbrückt hat |
| `issues.resolved` | Probleme, die irgendwann offen waren und am Ende weder offen noch akzeptiert sind, also durch Umformulierung behoben |
| `pm_messages` | Anzahl der Nachrichten des PM (inkl. Epic-Auswahl) |
| `seconds_total`, `seconds_per_turn_avg` | Antwortzeiten des Agenten (Summe / Durchschnitt) |

Ein Qualitätsproblem (`issue`) hat immer:
`field` (betroffenes Feld), `dimension` (`unambiguity`, `consistency`, `traceability`, `testability`),
`message` (Beschreibung), `text` (betroffener Feldinhalt) und `questions_asked` (wie oft der Agent
zu diesem Problem nachgefragt hat, höchstens 2).

### `bridged[]`: manuelle Überbrückungen

Der PM kann jedes offene Problem in der Oberfläche mit **„Keep as is“** bewusst stehen lassen. Das ist
der einzige Weg, auf dem ein Problem als überbrückt gilt, automatisch geschieht das nie. Eine Überbrückung gilt je Feld und
Dimension: Wer z. B. ein Eindeutigkeitsproblem der Benefit-Hypothese überbrückt, bekommt ein
Testbarkeitsproblem desselben Felds weiterhin gemeldet. Nach zwei
Rückfragen fragt der Agent zu einem Problem nicht mehr nach, es bleibt aber offen.

| Feld | Bedeutung |
|---|---|
| `time` | Zeitpunkt des Klicks |
| `after_pm_messages` | wie viele Nachrichten der PM bis dahin geschrieben hatte |
| `field`, `dimension`, `message`, `text` | das überbrückte Problem |
| `questions_asked` | wie oft der Agent bis dahin zu diesem Problem nachgefragt hatte |

## Hinweise zur Auswertung

- **Verbessert / überbrückt / offen** pro Dialog: Länge von `issues.resolved`, `issues.accepted` (= `bridged`), `issues.open`.
- **Rückfragen bis zum fertigen Feature**: `complete_after_pm_messages`.
- **Vollständigkeit**: `result.complete` und ob alle drei Felder in `result.ticket` gefüllt sind.
- **Dialoglänge**: `pm_messages`; **Antwortzeit**: `seconds_per_turn_avg`.
- **Verlauf eines Problems**: in `turns[]` nachsehen, ab welcher Runde es in `open_issues` auftaucht
  und wann es verschwindet oder in `accepted_issues` wandert.
- „Resolved“ heißt nur: Das Problem ist nach einer Umformulierung nicht mehr aufgetreten. Ob die neue
  Formulierung inhaltlich besser ist, muss bei der Evaluation selbst beurteilt werden.
- Ein Akzeptanzkriterium wird über seinen Text identifiziert. Wird es umformuliert, gilt ein altes
  Problem als behoben, und ein eventuelles neues Problem erscheint als neuer Eintrag.

Kurzer Überblick über alle Dialoge im Terminal (aus dem Projektordner):

```bash
.venv/bin/python -c "
import json, glob
for p in sorted(glob.glob('runs/*.json')):
    r = json.load(open(p))['result']; i = r['issues']
    print(p, 'complete' if r['complete'] else 'incomplete', 'msgs', r['pm_messages'],
          'open', len(i['open']), 'accepted', len(i['accepted']), 'resolved', len(i['resolved']),
          'avg_s', r['seconds_per_turn_avg'])
"
```
