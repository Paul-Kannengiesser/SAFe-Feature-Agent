"""Rule-based parts of the dialogue control (no LLM needed)."""

from feature_agent.agent import MAX_IMPROVEMENT_ROUNDS, Agent, _key
from feature_agent.quality import Finding
from feature_agent.schema import FeatureDraft, TracedText


def agent_with_issues() -> Agent:
    agent = Agent(retriever=None)
    agent.phase = "fields"
    agent.name_confirmed = True
    agent.draft = FeatureDraft(
        name=TracedText(value="ABC-XYZ | Title"),
        benefit_hypothesis=TracedText(value="Users save time."),
        acceptance_criteria=[TracedText(value="The page loads fast.")],
    )
    agent.findings = [
        Finding("benefit_hypothesis", "testability", "not measurable", "Users save time."),
        Finding("acceptance_criteria", "unambiguity", "vague: fast", "The page loads fast."),
    ]
    return agent


def test_issue_is_asked_at_most_twice_and_stays_open():
    agent = agent_with_issues()
    bh = agent.findings[0]
    assert agent.next_field() == "benefit_hypothesis"
    agent.issue_rounds[_key(bh)] = MAX_IMPROVEMENT_ROUNDS
    # the benefit issue is no longer asked about, but stays open (no automatic acceptance)
    assert agent.next_field() == "acceptance_criteria"
    assert bh in agent.findings and not agent.accepted


def test_rounds_are_counted_per_issue_not_per_field():
    agent = agent_with_issues()
    second = Finding("acceptance_criteria", "testability", "no condition", "The page loads fast.")
    agent.findings.append(second)
    agent.issue_rounds[_key(agent.findings[0])] = MAX_IMPROVEMENT_ROUNDS
    agent.issue_rounds[_key(agent.findings[1])] = MAX_IMPROVEMENT_ROUNDS
    assert agent._askable("acceptance_criteria") == [second]
    assert agent.next_field() == "acceptance_criteria"


def test_round_limit_holds_after_completion(monkeypatch):
    """After the completion message the agent never asks about a capped issue again,
    whatever the PM writes (there is no reset of the counters)."""
    import feature_agent.agent as agent_module
    from feature_agent.agent import TurnAnalysis

    agent = agent_with_issues()
    for finding in agent.findings:
        agent.issue_rounds[_key(finding)] = MAX_IMPROVEMENT_ROUNDS
    assert agent.next_field() is None  # all fields done, issues stay open
    findings = list(agent.findings)
    monkeypatch.setattr(agent_module, "check", lambda draft, turns, log=None: list(findings))
    monkeypatch.setattr(agent, "_analyse", lambda: TurnAnalysis(draft=agent.draft, agrees=False))
    monkeypatch.setattr(agent, "_respond", lambda field: "reply")
    agent.handle("Let's work on the open quality issues again.")
    assert agent.target is None and agent.next_field() is None
    assert all(agent.issue_rounds[_key(f)] == MAX_IMPROVEMENT_ROUNDS for f in findings)
    assert len(agent.findings) == 2 and not agent.accepted


def test_manual_bridging_is_logged_and_completes_the_feature():
    agent = agent_with_issues()
    for finding in list(agent.findings):
        agent.issue_rounds[_key(finding)] = 1
        agent.accept(finding)
    assert not agent.findings and len(agent.accepted) == 2
    assert [b["field"] for b in agent.bridged] == ["benefit_hypothesis", "acceptance_criteria"]
    assert agent.bridged[0]["questions_asked"] == 1 and agent.bridged[0]["after_pm_messages"] == 0
    assert agent.next_field() is None and agent.completed_after == 0


def test_bridging_applies_per_field_and_dimension(monkeypatch):
    """Bridging an unambiguity issue of the benefit hypothesis must not hide a testability
    issue of the same field; the bridged issue stays bridged after a rewording."""
    import feature_agent.agent as agent_module
    from feature_agent.agent import TurnAnalysis

    agent = agent_with_issues()
    bh_unambiguity = Finding("benefit_hypothesis", "unambiguity", "too short", "Users save time.")
    agent.findings = [bh_unambiguity]
    agent.accept(bh_unambiguity)
    reworded = "Users save a lot of time when they use the new export."
    later = [Finding("benefit_hypothesis", "unambiguity", "still vague", reworded),
             Finding("benefit_hypothesis", "testability", "no number", reworded)]
    monkeypatch.setattr(agent_module, "check", lambda draft, turns, log=None: list(later))
    monkeypatch.setattr(agent, "_analyse", lambda: TurnAnalysis(draft=agent.draft, agrees=False))
    monkeypatch.setattr(agent, "_respond", lambda field: "reply")
    agent.phase = "fields"
    agent.handle("I reworded it.")
    assert [(f.field, f.dimension) for f in agent.findings] == [("benefit_hypothesis", "testability")]


def test_skipped_llm_review_is_logged(monkeypatch):
    import feature_agent.quality as quality

    draft = FeatureDraft(
        benefit_hypothesis=TracedText(value="Reports are found 30 % faster by the 120 inspectors."),
        acceptance_criteria=[TracedText(value="When the user searches, results appear within 2 seconds.")],
    )
    monkeypatch.setattr(quality.llm, "structured", lambda *args, **kwargs: quality.Reviews(reviews=[]))
    log = {}
    assert quality.llm_checks(draft, skip=set(), log=log) == []
    assert log["llm_review_skipped"] is True
    assert log["llm_review_skipped_fields"] == ["acceptance_criteria", "benefit_hypothesis"]


def _agent_after_epic(*pm_messages: str) -> Agent:
    agent = Agent(retriever=None)
    agent.phase = "fields"
    for text in pm_messages:
        agent.messages.append(("user", text))
    return agent


def test_no_silent_prefill_of_benefit_hypothesis():
    """A benefit hypothesis derived from the feature idea (message 0) is not taken over
    while the PM answers something else (message 1); the name proposal is."""
    from feature_agent.schema import Provenance

    agent = _agent_after_epic("We need a print preview so inspectors print faster.", "1")
    result = FeatureDraft(
        name=TracedText(value="ABC-PRV | Print Preview", provenance=Provenance(user_turns=[0])),
        benefit_hypothesis=TracedText(value="Inspectors print faster.", provenance=Provenance(user_turns=[0])),
        acceptance_criteria=[TracedText(value="The preview opens.", provenance=Provenance(user_turns=[0]))],
    )
    agent._apply(result)
    assert agent.draft.name.value == "ABC-PRV | Print Preview"
    assert not agent.draft.is_filled("benefit_hypothesis")
    assert not agent.draft.is_filled("acceptance_criteria")


def test_unasked_field_is_taken_over_if_pm_just_said_it():
    from feature_agent.schema import Provenance

    agent = _agent_after_epic("We need a print preview.", "1", "Fine. It saves 20 % of reprints.")
    agent.target = "name"
    result = FeatureDraft(
        name=TracedText(value="ABC-PRV | Print Preview"),
        benefit_hypothesis=TracedText(value="Saves 20 % of reprints.", provenance=Provenance(user_turns=[2])),
    )
    agent._apply(result)
    assert agent.draft.benefit_hypothesis.value == "Saves 20 % of reprints."


def test_numbering_is_removed_from_name_unless_epic_or_pm_uses_it():
    agent = _agent_after_epic("We need a frontend.", "1")
    agent.sibling_names = ["ABC-SRV | Server", "ABC-PDF | PDF Rendering"]
    assert agent._strip_numbering("ABC-Frontend-01 | Preview") == "ABC-Frontend | Preview"
    assert agent._strip_numbering("ABC-FE | Preview") == "ABC-FE | Preview"
    agent.messages.append(("user", "Please call it ABC-FE-02."))
    assert agent._strip_numbering("ABC-FE-02 | Preview") == "ABC-FE-02 | Preview"
    agent.sibling_names = ["ABC-SRV-01 | Server"]
    assert agent._strip_numbering("ABC-Frontend-01 | Preview") == "ABC-Frontend-01 | Preview"


def test_confirmed_name_is_not_asked_again(monkeypatch):
    import feature_agent.agent as agent_module

    agent = agent_with_issues()
    agent.messages.append(("user", "Remove the -01."))
    sent = {}
    monkeypatch.setattr(agent_module.llm, "chat", lambda system, history, temperature: sent.update(task=history[-1][1]) or "ok")
    agent._respond("benefit_hypothesis")
    assert "Do not ask about the name again." in sent["task"]
