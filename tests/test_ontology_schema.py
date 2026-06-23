"""Ontology mock structural tests — isomorphism with the canonical data schema
and link integrity. These guard against metadata drifting from the real frames
(docs/poc_scope_v7.md §8; the mock is only useful if it stays AOS-isomorphic AND
data-faithful)."""

from __future__ import annotations

import pytest

from bakery.data.calendar import CALENDAR_DAILY_COLUMNS
from bakery.data.schema import DAILY_COLUMNS
from bakery.data.weather import WEATHER_DAILY_COLUMNS
from bakery.ontology.schema import (
    BACKING_CALENDAR,
    BACKING_DAILY,
    BACKING_WEATHER,
    BAKERY_ONTOLOGY,
)

_BACKING_COLUMNS = {
    BACKING_DAILY: set(DAILY_COLUMNS),
    BACKING_WEATHER: set(WEATHER_DAILY_COLUMNS),
    BACKING_CALENDAR: set(CALENDAR_DAILY_COLUMNS),
}


@pytest.mark.parametrize("obj", BAKERY_ONTOLOGY.objects, ids=lambda o: o.name)
def test_object_fields_subset_of_backing_schema(obj):
    """Every ontology field must exist as a column in its backing data schema."""
    allowed = _BACKING_COLUMNS[obj.backing]
    unknown = set(obj.field_names()) - allowed
    assert not unknown, f"{obj.name} has fields absent from {obj.backing}: {sorted(unknown)}"


@pytest.mark.parametrize("obj", BAKERY_ONTOLOGY.objects, ids=lambda o: o.name)
def test_every_object_has_primary_key(obj):
    assert obj.primary_key(), f"{obj.name} declares no primary key"


@pytest.mark.parametrize("link", BAKERY_ONTOLOGY.links, ids=lambda link_: link_.name)
def test_link_endpoints_exist(link):
    """Source/target must be real objects and join keys present on both sides."""
    source = BAKERY_ONTOLOGY.object(link.source)
    target = BAKERY_ONTOLOGY.object(link.target)
    for key in link.join_keys:
        assert key in source.field_names(), f"{link.name}: {key} missing on {source.name}"
        assert key in target.field_names(), f"{link.name}: {key} missing on {target.name}"


def test_cardinalities_are_recognized():
    valid = {"N:1", "1:N", "1:1"}
    bad = [link.name for link in BAKERY_ONTOLOGY.links if link.cardinality not in valid]
    assert not bad, f"unrecognized cardinality on: {bad}"


def test_lookup_helpers():
    assert BAKERY_ONTOLOGY.object("Item").name == "Item"
    assert any(link.target == "Category" for link in BAKERY_ONTOLOGY.links_from("Item"))
    assert any(link.source == "Store" for link in BAKERY_ONTOLOGY.links_to("DailySales"))
    with pytest.raises(KeyError):
        BAKERY_ONTOLOGY.object("Nonexistent")
