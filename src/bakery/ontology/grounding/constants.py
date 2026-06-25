"""Shared string constants for the grounding eval (grader types, frames).

Centralized so a typo is a NameError at import time, not a silent KeyError at
runtime. Values are the wire strings used in question definitions, output
schemas, and tool enums — do not change the string values.
"""

# grader_type values (Question.grader_type, scorer dispatch, OUTPUT_SCHEMAS keys)
NUMERIC = "numeric"
RANKING = "ranking"
DECOMPOSITION = "decomposition"
GRADER_TYPES = (NUMERIC, RANKING, DECOMPOSITION)

# frame values (demand_diff_by_condition: which DailyDataset frame to join)
CALENDAR = "calendar"
WEATHER = "weather"
FRAMES = (CALENDAR, WEATHER)
