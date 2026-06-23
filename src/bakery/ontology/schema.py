"""Bakery ontology mock — structurally isomorphic to AOS (docs/poc_scope_v7.md §8).

This is NOT a graph DB. Mirroring real AOS, it is a lightweight *semantic +
relational metadata layer* laid over the existing polyglot data (here: the
synthetic/real daily frames). Three layers, same shape as AOS:

  - OntologyObject + OntologyField   ← schema/meaning (displayName, description, PK)
  - OntologyLink                     ← relationships (join keys, cardinality = edges)
  - OntologyKnowledge                ← RAG concepts/formulae (semantics for the agent)

The point is not capability the bakery couldn't get from a plain join — at this
scale joins are simple. It is *structural isomorphism* with AOS so the with/without
delta we measure on this mock transfers to AOS at scale (docs §9, positioning).

Backing: each object names a `backing` frame key (one of DailyDataset's frames).
Field names are kept ⊆ the canonical data schema columns so the metadata never
drifts from the real data — `tests/test_ontology_schema.py` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass

# Backing frame keys — must match DailyDataset attribute names (data/loader.py).
BACKING_DAILY = "daily"
BACKING_WEATHER = "weather"
BACKING_CALENDAR = "calendar"


@dataclass(frozen=True)
class OntologyField:
    """One column's meaning — what the agent reads instead of guessing."""

    name: str               # must match the backing frame's column name
    type: str               # logical type (string/int/float/bool/datetime)
    description: str
    is_primary_key: bool = False
    is_name_key: bool = False


@dataclass(frozen=True)
class OntologyObject:
    """A typed object (≈ a table) with field-level meaning. AOS OntologyObject."""

    name: str
    display_name: str
    description: str
    backing: str                      # which DailyDataset frame holds the rows
    fields: tuple[OntologyField, ...]

    def primary_key(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.is_primary_key)

    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


@dataclass(frozen=True)
class OntologyLink:
    """A directed relationship between two objects — an edge with join semantics.

    cardinality is from source → target ("N:1" means many sources per target).
    join_keys are the column(s) equated on both sides (same names here).
    """

    name: str
    source: str            # OntologyObject.name
    target: str            # OntologyObject.name
    link_type: str         # belongs_to / sold_as / has / observed_on / had
    cardinality: str       # "N:1" | "1:N" | "1:1"
    join_keys: tuple[str, ...]


@dataclass(frozen=True)
class OntologyKnowledge:
    """A domain concept/formula chunk — the RAG layer's semantic supply."""

    name: str
    content: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Ontology:
    """The assembled metagraph: objects + links + knowledge, with lookup helpers."""

    objects: tuple[OntologyObject, ...]
    links: tuple[OntologyLink, ...]
    knowledge: tuple[OntologyKnowledge, ...] = ()

    def object(self, name: str) -> OntologyObject:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"unknown ontology object: {name}")

    def links_from(self, name: str) -> tuple[OntologyLink, ...]:
        return tuple(link for link in self.links if link.source == name)

    def links_to(self, name: str) -> tuple[OntologyLink, ...]:
        return tuple(link for link in self.links if link.target == name)


# --- Objects (isomorphic to data/schema.py, weather.py, calendar.py) ----------

_STORE = OntologyObject(
    name="Store", display_name="매장", backing=BACKING_DAILY,
    description="A bakery store (PoC: 광교 target + supporting stores).",
    fields=(
        OntologyField("store_id", "string", "Store identifier", is_primary_key=True, is_name_key=True),
    ),
)

_CATEGORY = OntologyObject(
    name="Category", display_name="카테고리", backing=BACKING_DAILY,
    description="Product category; demand is modeled at this grain first (Stage 1).",
    fields=(
        OntologyField("category_id", "string", "Category identifier", is_primary_key=True, is_name_key=True),
    ),
)

_ITEM = OntologyObject(
    name="Item", display_name="품목", backing=BACKING_DAILY,
    description="A sellable bakery product belonging to one category.",
    fields=(
        OntologyField("item_id", "string", "Item identifier", is_primary_key=True, is_name_key=True),
        OntologyField("category_id", "string", "Owning category (FK → Category)"),
    ),
)

_DAILY_SALES = OntologyObject(
    name="DailySales", display_name="일별 판매", backing=BACKING_DAILY,
    description="One (store, item, date) sales row. potential_demand is the "
                "censoring-corrected target; sold_units is the raw observation.",
    fields=(
        OntologyField("store_id", "string", "Store (FK → Store)", is_primary_key=True),
        OntologyField("item_id", "string", "Item (FK → Item)", is_primary_key=True),
        OntologyField("category_id", "string", "Category (FK → Category)"),
        OntologyField("date", "datetime", "Sales date (midnight-normalized)", is_primary_key=True),
        OntologyField("sold_units", "int", "Units sold (CENSORED on stockout days)"),
        OntologyField("is_stockout", "bool", "Whether the item stocked out that day"),
        OntologyField("stockout_time", "datetime", "Time of stockout (NaT if none)"),
        OntologyField("open_hours", "int", "Store open hours that day"),
        OntologyField("capacity", "int", "Daily production/stocking capacity"),
        OntologyField("potential_demand", "float", "Censoring-corrected true demand"),
    ),
)

_WEATHER = OntologyObject(
    name="Weather", display_name="날씨", backing=BACKING_WEATHER,
    description="Per-(store, date) daily weather. A demand driver fed to the forecast.",
    fields=(
        OntologyField("store_id", "string", "Store (FK → Store)", is_primary_key=True),
        OntologyField("date", "datetime", "Weather date", is_primary_key=True),
        OntologyField("avg_temp", "float", "Daily average temperature (°C)"),
        OntologyField("precipitation_mm", "float", "Precipitation (mm)"),
        OntologyField("is_rain", "int", "Rain flag (0/1)"),
    ),
)

_CALENDAR = OntologyObject(
    name="CalendarEvent", display_name="캘린더", backing=BACKING_CALENDAR,
    description="Per-date calendar/holiday features. Leakage-safe (depends only on date).",
    fields=(
        OntologyField("date", "datetime", "Calendar date", is_primary_key=True, is_name_key=True),
        OntologyField("holiday_name", "string", "Holiday name if any"),
        OntologyField("is_public_holiday", "int", "Public holiday flag (0/1)"),
        OntologyField("is_weekend", "int", "Weekend flag (0/1)"),
        OntologyField("is_off_day", "int", "Non-working day flag (0/1)"),
    ),
)

# StockoutEvent shares the daily frame but is its own object: censoring is a
# first-class concept (CLAUDE.md absolute rule #2), surfaced for the agent.
_STOCKOUT = OntologyObject(
    name="StockoutEvent", display_name="매진 이벤트", backing=BACKING_DAILY,
    description="A day an item stocked out. CENSORED: sold_units understates true "
                "demand. Risk modeling and sales modeling are kept separate.",
    fields=(
        OntologyField("store_id", "string", "Store (FK → Store)", is_primary_key=True),
        OntologyField("item_id", "string", "Item (FK → Item)", is_primary_key=True),
        OntologyField("date", "datetime", "Date of the stockout", is_primary_key=True),
        OntologyField("is_stockout", "bool", "True for stockout rows"),
        OntologyField("stockout_time", "datetime", "Time the item ran out"),
    ),
)


# --- Links (docs §8.2) --------------------------------------------------------

_LINKS = (
    OntologyLink("item_belongs_to_category", "Item", "Category", "belongs_to", "N:1", ("category_id",)),
    OntologyLink("item_sold_as_dailysales", "Item", "DailySales", "sold_as", "1:N", ("item_id",)),
    OntologyLink("store_has_dailysales", "Store", "DailySales", "has", "1:N", ("store_id",)),
    OntologyLink("dailysales_observed_on_weather", "DailySales", "Weather", "observed_on", "N:1", ("store_id", "date")),
    OntologyLink("dailysales_observed_on_calendar", "DailySales", "CalendarEvent", "observed_on", "N:1", ("date",)),
    OntologyLink("item_had_stockout", "Item", "StockoutEvent", "had", "1:N", ("item_id",)),
)


# --- Knowledge (RAG layer — domain definitions/formulae) ----------------------

_KNOWLEDGE = (
    OntologyKnowledge(
        "potential_demand",
        "potential_demand = censoring-corrected sold_units. On non-stockout days it "
        "equals sold_units; on early-stockout days it is uplifted to estimate true "
        "unmet demand. It, not sold_units, is the demand target.",
        tags=("censoring", "target", "demand"),
    ),
    OntologyKnowledge(
        "wape",
        "WAPE = Σ|actual−pred| / Σ|actual|. The primary accuracy metric; MAPE is "
        "banned because it explodes on zero/sparse items (CLAUDE.md absolute rule #5).",
        tags=("metric", "evaluation"),
    ),
    OntologyKnowledge(
        "order_policy",
        "Recommended order = demand_point × (1+safety_margin), then display-floor, then "
        "round up to batch unit. These are declared decision rules, not learned — which "
        "is what makes the decision lineage meaningful.",
        tags=("decision", "policy", "lineage"),
    ),
    OntologyKnowledge(
        "stockout_risk",
        "P(stockout) = P(demand > order) under demand ~ truncated Normal(point, cv·point). "
        "A Monte-Carlo placeholder distribution; sparse items can overstate P(stockout).",
        tags=("risk", "monte-carlo", "limitation"),
    ),
)


BAKERY_ONTOLOGY = Ontology(
    objects=(_STORE, _CATEGORY, _ITEM, _DAILY_SALES, _WEATHER, _CALENDAR, _STOCKOUT),
    links=_LINKS,
    knowledge=_KNOWLEDGE,
)

# Maps each backing key to the canonical schema column dict it must stay ⊆ of.
# Used by the isomorphism test; kept here so schema + ontology evolve together.
BACKING_SCHEMA_REF = {
    BACKING_DAILY: "bakery.data.schema.DAILY_COLUMNS",
    BACKING_WEATHER: "bakery.data.weather.WEATHER_DAILY_COLUMNS",
    BACKING_CALENDAR: "bakery.data.calendar.CALENDAR_DAILY_COLUMNS",
}
