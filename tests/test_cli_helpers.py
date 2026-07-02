"""Unit tests for cli.py pure helpers extracted from closed-loop / scenario-commit
commands (dedup refactor). No LLM, no dataset — just parsing + output plumbing."""

from __future__ import annotations

import pytest

from bakery.cli import _parse_period, _write_and_label
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
