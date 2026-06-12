"""Shared constants."""
from datetime import date

# The dataset's activity window ends in late May 2026. The reference
# "today" is frozen at the value used throughout development so that every
# reproduction of the ranking step is deterministic: re-running next month
# must not silently change recency scores or the submitted order.
REFERENCE_DATE = date(2026, 6, 11)
