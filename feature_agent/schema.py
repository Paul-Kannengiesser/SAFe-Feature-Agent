"""Target schema of the SAFe feature (building block 3); every value carries its provenance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FieldName = Literal["name", "benefit_hypothesis", "acceptance_criteria"]

# order in which the fields are elicited
FIELD_ORDER: tuple[FieldName, ...] = ("name", "benefit_hypothesis", "acceptance_criteria")

FIELD_LABELS: dict[FieldName, str] = {
    "name": "Name",
    "benefit_hypothesis": "Benefit Hypothesis",
    "acceptance_criteria": "Acceptance Criteria",
}


class Provenance(BaseModel):
    """Where a field value comes from."""

    user_turns: list[int] = Field(
        default_factory=list, description="Indices of the user messages the value is based on"
    )
    backlog_keys: list[str] = Field(
        default_factory=list, description="Jira keys of backlog items the value is based on"
    )


class TracedText(BaseModel):
    value: str = ""  # empty = not yet elicited
    provenance: Provenance = Field(default_factory=Provenance)


class FeatureDraft(BaseModel):
    """The feature under construction. Not nullable, otherwise the LLM tends to return null."""

    name: TracedText = Field(
        default_factory=TracedText,
        description="Short phrase that gives the feature a name and context",
    )
    benefit_hypothesis: TracedText = Field(
        default_factory=TracedText,
        description="The proposed measurable benefit to the end user or the business",
    )
    acceptance_criteria: list[TracedText] = Field(
        default_factory=list,
        description="Conditions under which the implementation counts as successful, one per entry",
    )

    def is_filled(self, field: FieldName) -> bool:
        if field == "acceptance_criteria":
            return any(ac.value.strip() for ac in self.acceptance_criteria)
        return bool(getattr(self, field).value.strip())

    def missing_fields(self) -> list[FieldName]:
        return [f for f in FIELD_ORDER if not self.is_filled(f)]

    def is_complete(self) -> bool:
        return not self.missing_fields()

    def field_text(self, field: FieldName) -> str:
        if field == "acceptance_criteria":
            return "\n".join(f"- {ac.value}" for ac in self.acceptance_criteria if ac.value.strip())
        return getattr(self, field).value.strip()
