"""Quality check of the field contents (building block 8).

Heuristics first, then NLI for consistency and traceability and the LLM for
unambiguity and testability.
"""

import os
import re
from dataclasses import dataclass
from functools import cache

from pydantic import BaseModel, Field

from . import llm
from .config import settings
from .schema import FeatureDraft, FieldName

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

VAGUE_WORDS = ["fast", "quick", "easy", "simple", "user-friendly", "intuitive", "appropriate",
               "flexible", "efficient", "better", "improved", "etc", "and/or", "as needed"]
NUMBER = re.compile(r"\d|%|\bpercent\b", re.I)  # a number makes the benefit measurable
TRACE_MIN_ENTAILMENT = 0.5
CONTRADICTION_MIN = 0.8


@dataclass
class Finding:
    field: FieldName
    dimension: str  # unambiguity, consistency, traceability, testability
    message: str
    text: str = ""  # the affected value


def check(draft: FeatureDraft, user_turns: list[str], log: dict | None = None) -> list[Finding]:
    """`log` receives notes for the run log, e.g. a skipped LLM review."""
    findings = heuristics(draft)
    already_flagged = {f.text for f in findings}
    findings += nli_checks(draft, user_turns)
    findings += llm_checks(draft, skip=already_flagged, log=log)
    return findings


def _items(draft: FeatureDraft):
    """(field, TracedText) for the benefit hypothesis and every acceptance criterion."""
    items = [("benefit_hypothesis", draft.benefit_hypothesis)]
    items += [("acceptance_criteria", ac) for ac in draft.acceptance_criteria]
    return [(field, item) for field, item in items if item.value.strip()]


def heuristics(draft: FeatureDraft) -> list[Finding]:
    findings = []
    if len(draft.name.value) > 80:
        findings.append(Finding("name", "unambiguity", "The name is too long for a short phrase.",
                                draft.name.value))

    bh = draft.benefit_hypothesis.value
    if bh and len(bh.split()) < 8:
        findings.append(Finding("benefit_hypothesis", "unambiguity",
                                "The benefit hypothesis is too short to be clear.", bh))
    elif bh and not NUMBER.search(bh):
        findings.append(Finding("benefit_hypothesis", "testability",
                                "The benefit is not measurable (no number, e.g. an expected change in %).", bh))

    for field, ac in _items(draft):
        vague = [w for w in VAGUE_WORDS if re.search(rf"\b{re.escape(w)}\b", ac.value, re.I)]
        if field == "acceptance_criteria" and vague:
            findings.append(Finding(field, "unambiguity",
                                    f"'{ac.value}' contains vague wording: {', '.join(vague)}.", ac.value))
    return findings


@cache
def _nli_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model).eval().to(device)
    return tokenizer, model, device


def nli(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Probabilities of entailment / neutral / contradiction for (premise, hypothesis) pairs."""
    if not pairs:
        return []
    import torch

    tokenizer, model, device = _nli_model()
    with torch.no_grad():
        encoded = tokenizer([p for p, _ in pairs], [h for _, h in pairs], return_tensors="pt",
                            padding=True, truncation=True).to(device)
        probs = model(**encoded).logits.softmax(-1).cpu().tolist()
    labels = model.config.id2label
    return [{labels[i]: p for i, p in enumerate(row)} for row in probs]


def nli_checks(draft: FeatureDraft, user_turns: list[str]) -> list[Finding]:
    findings = []
    items = _items(draft)

    # Traceability: best entailment over the single PM messages, the provenance messages and all together.
    # Both sides are phrased as requirements, otherwise NLI rates "must work offline" -> "works offline" neutral.
    for field, item in items:
        named = [user_turns[i] for i in item.provenance.user_turns if 0 <= i < len(user_turns)]
        premises = user_turns + [" ".join(named), " ".join(user_turns)]
        value = item.value.strip()
        hypothesis = f"The product manager requires that {value[0].lower()}{value[1:]}"
        pairs = [("The product manager requires the following: " + p, hypothesis) for p in premises if p]
        if max(probs["entailment"] for probs in nli(pairs)) < TRACE_MIN_ENTAILMENT:
            findings.append(Finding(field, "traceability",
                                    f"The agent wrote '{item.value}', but this goes beyond what the PM said. "
                                    "The PM should confirm or correct it.", item.value))

    # Consistency: pairwise, in both directions
    texts = [item.value for _, item in items]
    pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
    reverse = [(b, a) for a, b in pairs]
    for (a, b), p1, p2 in zip(pairs, nli(pairs), nli(reverse)):
        if max(p1["contradiction"], p2["contradiction"]) >= CONTRADICTION_MIN:
            findings.append(Finding("acceptance_criteria", "consistency",
                                    f"'{b}' contradicts '{a}'.", b))
    return findings


class ItemReview(BaseModel):
    unambiguous: bool = Field(description="Only one reasonable interpretation, no vague words")
    testable: bool = Field(description="Can be objectively checked whether it is fulfilled")
    problem: str = Field(description="If not unambiguous or not testable: the problem in max. 15 words, else empty")


class Reviews(BaseModel):
    reviews: list[ItemReview]


REVIEW_PROMPT = """You review parts of a SAFe Feature (benefit hypothesis, acceptance criteria) for quality.
Return one review per numbered item, in the same order.
Be strict but fair: an acceptance criterion is testable if a tester could decide pass/fail without
asking anyone. It does not need to start with 'when'."""

_review_cache: dict[tuple[str, str], ItemReview] = {}


def llm_checks(draft: FeatureDraft, skip: set[str], log: dict | None = None) -> list[Finding]:
    items = [(field, item.value) for field, item in _items(draft) if item.value not in skip]
    new = [item for item in items if item not in _review_cache]
    if new:
        listing = "\n".join(f"{n}. ({field.replace('_', ' ')}) {text}"
                            for n, (field, text) in enumerate(new, 1))
        reviews = llm.structured(REVIEW_PROMPT, listing, Reviews).reviews
        if len(reviews) == len(new):
            _review_cache.update(zip(new, reviews))
        elif log is not None:
            # wrong number of reviews: skip this round, the texts are reviewed again next turn
            log["llm_review_skipped"] = True
            log["llm_review_skipped_fields"] = sorted({field for field, _ in new})
            log["llm_review_skipped_texts"] = [text for _, text in new]

    findings = []
    for field, text in items:
        review = _review_cache.get((field, text))
        problem = review.problem.split(". ")[0].rstrip(".") + "." if review else ""
        if review and not review.unambiguous:
            findings.append(Finding(field, "unambiguity", f"'{text}': {problem}", text))
        elif review and not review.testable:
            findings.append(Finding(field, "testability", f"'{text}': {problem}", text))
    return findings
