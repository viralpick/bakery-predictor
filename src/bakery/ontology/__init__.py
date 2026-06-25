"""Bakery ontology mock (v7 LayerB) — AOS-isomorphic semantic/relational layer.

See docs/poc_scope_v7.md §8. Self-contained: objects + links + knowledge over the
existing daily frames (schema.py), plus OntologyFunctions wrapping the v6 decision
layer (functions.py). The forecast core stays untouched; this layer is read-only.
"""

from .functions import (
    FUNCTION_REGISTRY,
    OntologyFunctionSpec,
    WhatIfResult,
    demand_diff_by_condition,
    explain_order,
    rank_stockout_risk,
    waste_cost,
    what_if,
)
from .schema import (
    BACKING_SCHEMA_REF,
    BAKERY_ONTOLOGY,
    Ontology,
    OntologyField,
    OntologyKnowledge,
    OntologyLink,
    OntologyObject,
)
from .writeback import OrderRecord, WritebackStore

__all__ = [
    "BAKERY_ONTOLOGY", "Ontology", "OntologyObject", "OntologyField",
    "OntologyLink", "OntologyKnowledge", "BACKING_SCHEMA_REF",
    "FUNCTION_REGISTRY", "OntologyFunctionSpec", "WhatIfResult",
    "rank_stockout_risk", "explain_order", "what_if", "waste_cost",
    "demand_diff_by_condition",
    "OrderRecord", "WritebackStore",
]
