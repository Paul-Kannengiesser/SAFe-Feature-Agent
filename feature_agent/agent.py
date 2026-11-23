"""Dialogue control (building blocks 4-7).

Each slot-filling turn: analyse (LLM) -> check -> policy (rules) -> respond (LLM).
"""

import re
import time
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from . import llm
from .quality import Finding, check
from .retrieval import Retriever
from .schema import FIELD_LABELS, FIELD_ORDER, FeatureDraft, FieldName
from .sources import IssueFilter

MAX_IMPROVEMENT_ROUNDS = 2  # questions per quality issue and for the name
OVERLAP_MIN = 0.7  # cosine similarity for the duplicate hint
REPEAT_MIN = 0.8  # text similarity at which a reply counts as a repetition

FEATURE_DEFINITION = """A SAFe Feature has exactly three fields:
- Name: a short phrase that gives the feature a name and context.
- Benefit Hypothesis: the proposed measurable benefit to the end user or the business.
- Acceptance Criteria: conditions under which the implementation counts as successful."""

ANALYSE_PROMPT = f"""You maintain the draft of a new SAFe Feature based on a conversation with a Product Manager.
{FEATURE_DEFINITION}

Return the complete updated draft and the Product Manager's intent.
Draft rules:
- Fill a field only with information the Product Manager gave. Never invent content and never copy content from existing backlog items.
- Agent messages are only context. Wording or content that appears only in an Agent message must not be used, unless the Product Manager's reply explicitly agrees with it (e.g. "yes", "sounds good", "take that"). Ignoring the suggestion, answering something else or "I just told you" means NO.
- Acceptance criteria: only conditions the Product Manager stated. Exactly one condition per entry. If the Product Manager clarifies or improves an existing criterion, replace it instead of adding a new one. Explanations (e.g. how something can be tested) are not new criteria; use them to make the related criterion more precise.
- The Product Manager's last message may contain information for several fields. Put every piece of information into the field it belongs to, even if the last question was about another field.
- Keep existing values unless the Product Manager changed, improved or added to them; then rewrite the value so that it includes the new information.
- The name must follow exactly the naming pattern of the existing features of the epic, if there is one (e.g. "ABC-XYZ | Short Title": a short code, a pipe, a title). Do not add numbers, versions or other suffixes that the existing names do not have, and never use a Jira key (e.g. "ABC-123") in the name. The name must differ from the names of existing features.
- provenance.user_turns: numbers of the Product Manager messages the value is based on.
- provenance.backlog_keys: Jira keys of backlog items the value is based on (e.g. the naming pattern), otherwise empty.
Intent:
- agrees: true if the Product Manager agrees with the current value of the field under discussion (e.g. "yes", "fine", "keep it")."""

RESPOND_PROMPT = f"""You are a helpful assistant who helps a Product Manager define a new SAFe Feature in a natural conversation.
{FEATURE_DEFINITION}

Write your next message in English (max. 80 words):
1. First react to the Product Manager's last message in one sentence: acknowledge what they said, answer their question, or apologise briefly if they say you repeated yourself.
2. Then do your task for this message:
   - If a field is missing, ask for it.
   - If a field has open quality issues, explain the most important issue in plain words and ask how to improve it. You may make a concrete suggestion, but always mark it as a suggestion the Product Manager can accept or change.
   - If the name was proposed by you, show it and ask the Product Manager to confirm or change it.
   - If all fields are done, say so and honestly mention open issues if there are any. The Product Manager can
     still rephrase a field or keep an issue with "Keep as is".
     There is no submit function: the Product Manager copies the fields from the ticket on the right into Jira.
   - If all fields were already done before, only answer the Product Manager's last message.
Rules:
- Use the backlog context: if an existing feature (in the same epic or another epic) clearly overlaps with
  the new feature, point this out once, name its key and name, and ask how the new feature differs.
  Do not do this for features that are only loosely related.
- Only describe changes that are listed under "Changes in this turn". Never claim that an issue is resolved while it is still listed as open.
- Never bring up issues the Product Manager accepted as they are, and do not repeat suggestions they rejected.
- If the Product Manager wants to keep a wording although an issue is open, tell them they can click
  "Keep as is" next to the issue in the list on the right. You cannot mark issues as kept yourself.
- Never repeat one of your previous messages.
- Never invent facts about the feature.
- Do not repeat a question you already asked; if the Product Manager could not answer it, help with an example or move on.
- Vary your wording; do not start like your previous messages. Never greet (no "Hi", "Hello").
- Never mention the state, its labels or your task; just write the message."""

ORDINALS = {"first": 1, "second": 2, "third": 3, "1st": 1, "2nd": 2, "3rd": 3}


class TurnAnalysis(BaseModel):
    draft: FeatureDraft
    agrees: bool = Field(description="PM agrees with the current value of the discussed field")


def _key(finding: Finding) -> tuple[str, str, str]:
    """Issue identity for bridging and question counting: field and dimension,
    plus the text for acceptance criteria (there can be several)."""
    text = finding.text if finding.field == "acceptance_criteria" else ""
    return finding.field, finding.dimension, text


class Agent:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.messages: list[tuple[str, str]] = []  # (role, text), role = "user" | "assistant"
        self.phase = "idea"  # idea -> epic -> fields
        self.epic = None
        self.epic_candidates = []
        self.context = ""
        self.draft = FeatureDraft()
        self.findings: list[Finding] = []
        self.accepted: list[Finding] = []  # bridged via "Keep as is"
        self.issue_rounds: dict[tuple, int] = {}  # questions asked per issue
        self.target: FieldName | None = None  # field the last message was about
        self.name_confirmed = False
        self.name_rounds = 0
        self.completion_announced = False
        self.overlaps = []  # possible duplicates
        self.overlap_mentioned = False
        self.sibling_names: list[str] = []  # features of the chosen epic
        self.changes: list[str] = []
        # run log (export.py)
        self.started = datetime.now()
        self.turns: list[dict] = []
        self.seen: dict[tuple, Finding] = {}  # every issue that was ever open
        self.bridged: list[dict] = []
        self.completed_after: int | None = None  # PM messages until all fields were done
        self.intent: dict | None = None
        self.review_log: dict = {}
        self._say("Hi! Describe the feature you have in mind in a few sentences.")

    def handle(self, text: str) -> None:
        start = time.time()
        self.messages.append(("user", text))
        self.changes, self.intent, self.review_log = [], None, {}
        if self.phase == "idea":
            self._propose_epics(text)
        elif self.phase == "epic":
            self._choose_epic(text)
        else:
            self._turn()
        self.turns.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "pm": text,
            "agent": self.messages[-1][1],
            "seconds": round(time.time() - start, 1),
            "phase": self.phase,
            "target": self.target,
            "intent": self.intent,
            "llm_review_skipped": self.review_log.get("llm_review_skipped", False),
            "llm_review_skipped_fields": self.review_log.get("llm_review_skipped_fields", []),
            "changes": list(self.changes),
            "ticket": {f: self.draft.field_text(f) for f in FIELD_ORDER},
            "open_issues": self.issue_dicts(self.findings),
            "accepted_issues": self.issue_dicts(self.accepted),
        })
        self._note_completion()

    def accept(self, finding: Finding) -> None:
        """Manual bridging via the "Keep as is" button, never automatic."""
        if finding not in self.findings:
            return
        self.findings.remove(finding)
        self.accepted.append(finding)
        self.bridged.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "after_pm_messages": len(self.turns),
            **self.issue_dicts([finding])[0],
        })
        self._note_completion()

    def issue_dicts(self, findings: list[Finding]) -> list[dict]:
        return [{**asdict(f), "questions_asked": self.issue_rounds.get(_key(f), 0)} for f in findings]

    def _note_completion(self) -> None:
        if self.completed_after is None and self.is_done():
            self.completed_after = len(self.turns)

    def _say(self, text: str) -> None:
        self.messages.append(("assistant", text))

    def user_turns(self) -> list[str]:
        return [text for role, text in self.messages if role == "user"]

    def is_done(self) -> bool:
        return self.phase == "fields" and self.next_field() is None

    def _propose_epics(self, idea: str) -> None:
        self.epic_candidates = [h.item for h in self.retriever.similar(idea, k=3, issue_type="Epic")]
        options = "\n".join(f"{n}. {e.key}: {e.name}" for n, e in enumerate(self.epic_candidates, 1))
        self._say(f"Which epic does this feature belong to?\n\n{options}\n\n"
                  "Answer with the number, another epic key, or 'none'.")
        self.phase = "epic"

    def _choose_epic(self, answer: str) -> None:
        key = re.search(r"[A-Z][A-Z0-9]+-\d+", answer.upper())
        words = re.findall(r"\w+", answer.lower())
        number = next((int(w) if w.isdigit() else ORDINALS[w]
                       for w in words if w.isdigit() or w in ORDINALS), None)
        if key and (item := self.retriever.get(key.group())) and item.issue_type == "Epic":
            self.epic = item
        elif number and 1 <= number <= len(self.epic_candidates):
            self.epic = self.epic_candidates[number - 1]
        elif "none" not in words:
            self._say("Sorry, I did not get that. Please answer with a number, an epic key or 'none'.")
            return
        self.phase = "fields"
        self.context = self._load_context()
        self._turn()

    def _load_context(self) -> str:
        """Backlog context for the LLM: epic, its features and similar features elsewhere."""
        idea = self.user_turns()[0]
        lines = []
        if self.epic:
            lines.append(f"Epic: {self.epic.header()}")
            siblings = self.retriever.structured(IssueFilter(issue_type="Feature", epic_key=self.epic.key))
            self.sibling_names = [f.name for f in siblings]
            lines.append("Existing features of this epic:")
            lines += [f"- {f.header()}" for f in siblings[:15]] or ["- none"]
            if siblings:
                lines.append("Naming pattern (names of these features): "
                             + "; ".join(f'"{f.name}"' for f in siblings[:8]))
        self.overlaps = [h for h in self.retriever.similar(idea, k=3) if h.similarity >= OVERLAP_MIN]
        similar = self.retriever.similar(idea, k=3, exclude_epic=self.epic.key if self.epic else None)
        lines.append("Most similar features in other epics (cosine similarity to the feature idea, 1.0 = identical):")
        for hit in similar:
            line = f"- {hit.item.header()} (similarity {hit.similarity:.2f})"
            if hit.item.benefit_hypothesis:
                line += f"\n  Benefit Hypothesis: {hit.item.benefit_hypothesis[:300]}"
            lines.append(line)
        return "\n".join(lines)

    def _turn(self) -> None:
        before = {f: self.draft.field_text(f) for f in FIELD_ORDER}
        analysis = self._analyse()
        self.intent = {"agrees": analysis.agrees}
        self._apply(analysis.draft)
        for field in FIELD_ORDER:
            if self.draft.field_text(field) != before[field]:
                self.changes.append(f"{FIELD_LABELS[field]} was set to: {self.draft.field_text(field)}")
        if self.target == "name" and (analysis.agrees or self.draft.name.value != before["name"]):
            self.name_confirmed = True
            self.changes.append("The PM confirmed the name.")

        self.findings = [f for f in check(self.draft, self.user_turns(), log=self.review_log)
                         if _key(f) not in {_key(a) for a in self.accepted}]
        self.seen.update({_key(f): f for f in self.findings})

        field = self.next_field()
        if field == "name" and self.draft.is_filled(field) and not self._askable(field):
            self.name_rounds += 1
        for finding in self._askable(field) if field else []:
            self.issue_rounds[_key(finding)] = self.issue_rounds.get(_key(finding), 0) + 1
        self.target = field
        self._say(self._respond(field))

    def _analyse(self) -> TurnAnalysis:
        about = FIELD_LABELS[self.target] if self.target else "the feature as a whole"
        prompt = (f"Backlog context:\n{self.context}\n\n"
                  f"Current draft:\n{self.draft.model_dump_json(indent=1)}\n\n"
                  f"Open quality issues:\n{self._issue_list(self.findings)}\n\n"
                  f"Conversation:\n{self._conversation()}\n\n"
                  f"Field under discussion: {about}")
        return llm.structured(ANALYSE_PROMPT, prompt, TurnAnalysis)

    def _apply(self, result: FeatureDraft) -> None:
        """Field-wise update. An empty field other than the name is only filled if it was
        asked for or comes from the PM's latest message, so the LLM cannot fill it silently."""
        latest = len(self.user_turns()) - 1
        for field in FIELD_ORDER:
            if field == "name" or self.draft.is_filled(field) or field == self.target:
                setattr(self.draft, field, getattr(result, field))
            elif field == "benefit_hypothesis":
                if latest in result.benefit_hypothesis.provenance.user_turns:
                    self.draft.benefit_hypothesis = result.benefit_hypothesis
            else:  # acceptance criteria: only those the PM just stated
                self.draft.acceptance_criteria = [ac for ac in result.acceptance_criteria
                                                  if latest in ac.provenance.user_turns]
        self.draft.name.value = self._strip_numbering(self.draft.name.value)

    def _strip_numbering(self, name: str) -> str:
        """Remove a numbering such as "-01" from the code part of the name ("ABC-01 | Title")
        if no feature of the epic has one and the PM did not ask for it."""
        code, sep, title = name.partition("|")
        numbers = re.findall(r"-\d+\b", code)
        if not sep or not numbers:
            return name
        if any(re.search(r"-\d+\b", sibling.partition("|")[0]) for sibling in self.sibling_names):
            return name
        if any(number in turn for number in numbers for turn in self.user_turns()):
            return name
        return re.sub(r"-\d+\b", "", code).rstrip() + " " + sep + title

    def next_field(self) -> FieldName | None:
        """Rule-based policy: first field that is missing, has an askable issue or an unconfirmed name."""
        for field in FIELD_ORDER:
            if not self.draft.is_filled(field) or self._askable(field):
                return field
            if field == "name" and not self.name_confirmed and self.name_rounds < MAX_IMPROVEMENT_ROUNDS:
                return field
        return None

    def findings_for(self, field: FieldName) -> list[Finding]:
        return [f for f in self.findings if f.field == field]

    def _askable(self, field: FieldName) -> list[Finding]:
        """Open issues of a field asked about fewer than MAX_IMPROVEMENT_ROUNDS times."""
        return [f for f in self.findings_for(field)
                if self.issue_rounds.get(_key(f), 0) < MAX_IMPROVEMENT_ROUNDS]

    def _respond(self, field: FieldName | None) -> str:
        if field is None and self.completion_announced:
            next_step = ("All fields were already done and you already said so. Do not repeat that; "
                         "only answer the PM's last message (e.g. a question or a change).")
        elif field is None:
            next_step = "All fields are done."
            if self.findings:
                next_step += (" Some issues are still open. Mention that the PM can keep them with the "
                              "'Keep as is' button next to each issue on the right, or rephrase the field.")
        elif not self.draft.is_filled(field):
            next_step = f"Ask for the missing field: {FIELD_LABELS[field]}"
        elif field == "name" and not self._askable(field):
            next_step = (f"You proposed the name '{self.draft.name.value}' based on the feature idea and the "
                         "naming pattern of the epic. Ask the PM to confirm or change it.")
        else:
            next_step = (f"Work on the open issues of {FIELD_LABELS[field]}:\n"
                         f"{self._issue_list(self._askable(field))}")
        self.completion_announced = field is None
        if self.name_confirmed and field != "name":
            next_step += " The PM has already confirmed the name. Do not ask about the name again."
        if self.overlaps and not self.overlap_mentioned:
            self.overlap_mentioned = True
            found = ", ".join(f"{h.item.key} '{h.item.name}' (similarity {h.similarity:.2f})" for h in self.overlaps)
            next_step = (f"First point out that the existing feature(s) {found} look very similar to the new "
                         f"feature (possible duplicate) and ask how the new feature differs. Then: {next_step}")
        state = (f"\n\n## Current state (not visible to the Product Manager)\n"
                 f"Current ticket:\n{self._ticket()}\n\n"
                 f"Open quality issues:\n{self._issue_list(self.findings)}\n\n"
                 f"Issues the PM bridged with 'Keep as is': {len(self.accepted)} (settled, do not mention them)\n\n"
                 f"Changes in this turn:\n" + ("\n".join(f"- {c}" for c in self.changes) or "none") + "\n\n"
                 f"Backlog context:\n{self.context}")
        # The task goes into the last PM message: the Qwen3 template in Ollama drops later system messages.
        history = self.messages[:-1] + [(
            "user", f"{self.messages[-1][1]}\n\n[Note for the assistant, not written by the PM - "
                    f"your task for this message: {next_step}]")]
        reply = llm.chat(RESPOND_PROMPT + state, history=history, temperature=0.6)
        if self._repeats(reply):
            history[-1] = (history[-1][0], history[-1][1] + f"\n[Your first draft repeated an earlier "
                           f"message: '{reply}'. Write a different, short reply.]")
            reply = llm.chat(RESPOND_PROMPT + state, history=history, temperature=0.9)
        return reply

    def _repeats(self, reply: str) -> bool:
        earlier = [text for role, text in self.messages if role == "assistant"]
        return any(SequenceMatcher(None, reply.lower(), text.lower()).ratio() >= REPEAT_MIN
                   for text in earlier[-3:])

    def _conversation(self) -> str:
        lines, turn = [], 0
        for role, text in self.messages:
            if role == "user":
                lines.append(f"[Product Manager message {turn}] {text}")
                turn += 1
            else:
                lines.append(f"[Agent] {text}")
        return "\n".join(lines)

    def _ticket(self) -> str:
        return "\n".join(f"{FIELD_LABELS[f]}:\n{self.draft.field_text(f) or '(missing)'}" for f in FIELD_ORDER)

    @staticmethod
    def _issue_list(findings: list[Finding]) -> str:
        return "\n".join(f"- {FIELD_LABELS[f.field]} ({f.dimension}): {f.message}" for f in findings) or "none"
