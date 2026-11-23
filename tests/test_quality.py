from feature_agent.quality import heuristics, nli_checks
from feature_agent.schema import FeatureDraft, Provenance, TracedText


def traced(value, *turns):
    return TracedText(value=value, provenance=Provenance(user_turns=list(turns)))


def test_heuristics_flag_unmeasurable_benefit_and_vague_criterion():
    draft = FeatureDraft(
        benefit_hypothesis=traced("Inspectors save time because they can preview documents before printing."),
        acceptance_criteria=[traced("The report can be downloaded fast."),
                             traced("When the user clicks print, the preview opens within 2 seconds.")],
    )
    found = {(f.field, f.dimension) for f in heuristics(draft)}
    assert found == {("benefit_hypothesis", "testability"), ("acceptance_criteria", "unambiguity")}


def test_nli_traceability_and_consistency():
    turns = ["The preview must work offline and the settings must be saved."]
    draft = FeatureDraft(acceptance_criteria=[
        traced("The preview works offline.", 0),
        traced("Each entry shows date, type and a download link.", 0),  # invented
        traced("The preview is only available when the device is online.", 0),  # contradiction
    ])
    found = {(f.dimension, f.text) for f in nli_checks(draft, turns)}
    assert ("traceability", "Each entry shows date, type and a download link.") in found
    assert ("traceability", "The preview works offline.") not in found
    assert ("consistency", "The preview is only available when the device is online.") in found
