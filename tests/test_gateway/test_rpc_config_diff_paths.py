"""Unit tests for ``rpc_config.diff_paths`` — the structural dotted-path diff
that lets YAML-mode ``config.apply`` persist only the fields the operator
actually changed versus the baseline they were shown.
"""

from __future__ import annotations

from agentos.gateway.rpc_config import diff_paths


def test_identical_dicts_have_no_diff():
    d = {"a": 1, "b": {"c": 2}}
    assert diff_paths(d, dict(d)) == set()


def test_changed_leaf_yields_its_dotted_path():
    old = {"debug": False, "auth": {"mode": "none"}}
    new = {"debug": False, "auth": {"mode": "token"}}
    assert diff_paths(old, new) == {"auth.mode"}


def test_added_key_yields_path_and_nested_leaves():
    old = {"a": 1}
    new = {"a": 1, "auth": {"mode": "token", "token": "x"}}
    assert diff_paths(old, new) == {"auth", "auth.mode", "auth.token"}


def test_removed_key_yields_path_and_nested_leaves():
    old = {"a": 1, "section": {"x": 1, "y": 2}}
    new = {"a": 1}
    assert diff_paths(old, new) == {"section", "section.x", "section.y"}


def test_changed_list_yields_list_path_not_indices():
    old = {"tiers": [1, 2, 3]}
    new = {"tiers": [1, 2, 4]}
    assert diff_paths(old, new) == {"tiers"}


def test_multiple_independent_changes():
    old = {"host": "127.0.0.1", "debug": False, "auth": {"mode": "none"}}
    new = {"host": "0.0.0.0", "debug": True, "auth": {"mode": "none"}}
    assert diff_paths(old, new) == {"host", "debug"}


def test_empty_baseline_marks_everything_changed():
    new = {"a": 1, "b": {"c": 2}}
    assert diff_paths({}, new) == {"a", "b", "b.c"}
