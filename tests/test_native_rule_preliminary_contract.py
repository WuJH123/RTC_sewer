from __future__ import annotations

from sewerrtc.control.native_rule_audit import extract_action_clause_references, preliminary_text_reference


def test_id_only_in_if_condition_is_not_confirmed_action_clause() -> None:
    text = """
    RULE R1
    IF LINK ADD301.2 FLOW > 1
    THEN ORIFICE cc006.1 SETTING = 0.5
    """
    refs = extract_action_clause_references(text, ["ADD301.2", "cc006.1"])
    assert [ref.object_id for ref in refs] == ["cc006.1"]


def test_id_in_then_action_clause_is_confirmed() -> None:
    text = """
    RULE R1
    IF NODE N1 DEPTH > 1
    THEN PUMP ADD301.2 STATUS = ON
    PRIORITY 2
    """
    refs = extract_action_clause_references(text, ["ADD301.2"])
    assert len(refs) == 1
    assert refs[0].clause == "THEN"


def test_similar_substring_id_does_not_match() -> None:
    text = "THEN PUMP ADD301.20 STATUS = ON"
    assert not preliminary_text_reference(text, "ADD301.2")
    assert extract_action_clause_references(text, ["ADD301.2"]) == []


def test_comment_id_does_not_count() -> None:
    text = "; THEN PUMP ADD301.2 STATUS = ON"
    assert not preliminary_text_reference(text, "ADD301.2")


def test_preliminary_field_cannot_unlock_downstream_stage() -> None:
    assert preliminary_text_reference("IF LINK ADD301.2 FLOW > 0", "ADD301.2")
    assert extract_action_clause_references("IF LINK ADD301.2 FLOW > 0", ["ADD301.2"]) == []
