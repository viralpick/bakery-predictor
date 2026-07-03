"""Unit tests for cli.py pure helpers extracted from closed-loop / scenario-commit
commands (dedup refactor). No LLM, no dataset — just parsing + output plumbing."""

from __future__ import annotations

import pytest

from bakery.cli import _lever_warning, _parse_period, _write_and_label
from bakery.ontology.writeback import WritebackStore


def test_parse_period_splits_and_strips():
    start, end, stamp = _parse_period(" 2026-07-06 , 2026-07-12 ", now="")
    assert start == "2026-07-06"
    assert end == "2026-07-12"
    assert stamp == "2026-07-06T09:00:00"          # default = start 09:00


def test_parse_period_honors_explicit_now():
    start, end, stamp = _parse_period("2026-07-06,2026-07-12", now="2026-07-06T13:30:00")
    assert stamp == "2026-07-06T13:30:00"          # explicit now wins


def test_write_and_label_writes_parquet_when_out(tmp_path, capsys):
    wb = WritebackStore(require_approval=True)
    out = tmp_path / "sc.parquet"
    _write_and_label(wb, str(out), source="synthetic")
    assert out.exists()
    printed = capsys.readouterr().out
    assert "wrote" in printed
    assert "mechanism demo" in printed             # synthetic label


def test_write_and_label_skips_parquet_when_no_out(tmp_path, capsys):
    wb = WritebackStore(require_approval=True)
    _write_and_label(wb, "", source="real")
    printed = capsys.readouterr().out
    assert "wrote" not in printed
    assert "source=real" in printed                # non-synthetic label


def test_lever_warning_fires_on_zero_baseline():
    warn = _lever_warning(0.0)
    assert warn is not None
    assert "before_demand" in warn                 # names the collapsed baseline


def test_lever_warning_silent_on_positive_baseline():
    assert _lever_warning(42.0) is None


def test_scenario_commit_batch_command_registered():
    import typer
    import bakery.cli as c
    group = typer.main.get_group(c.app)
    cmd = group.get_command(None, "scenario-commit-batch")
    assert cmd is not None
    opts = [p.name for p in cmd.params]
    assert "items" in opts
    assert "gate" in opts
    assert "policy" not in opts


def test_demand_absorption_command_registered():
    import typer
    import bakery.cli as c
    group = typer.main.get_group(c.app)
    cmd = group.get_command(None, "demand-absorption")
    assert cmd is not None
    opts = [p.name for p in cmd.params]
    assert "source" in opts
