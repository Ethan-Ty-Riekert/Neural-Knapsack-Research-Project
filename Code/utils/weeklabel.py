"""weeklabel.py - S2W<n> project-week labels.

Single source of truth for the week-numbering convention documented in
CLAUDE.md ("Week labels (S2W<n>)"): Monday-Sunday weeks, S2W1 = Monday
2026-07-20. Used to tag saved training/eval results so their age is readable
without cross-referencing dates by hand.
"""

from datetime import date

S2W1_MONDAY = date(2026, 7, 20)


def week_label(d: date | None = None) -> str:
    """Return the S2W<n> label for the given date (default: today)."""
    d = d or date.today()
    n = (d - S2W1_MONDAY).days // 7 + 1
    return f"S2W{n}"
