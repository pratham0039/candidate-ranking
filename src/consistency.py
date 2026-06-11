"""
Profile consistency checks (the "skeptical recruiter" layer).

Every claim a candidate makes should be possible given the rest of their
profile. A resume whose dates and durations do not add up cannot be trusted,
so profiles with hard internal impossibilities are excluded from ranking
before any scoring happens. This mirrors what a careful human recruiter does
on a first read.

Checks:
  1. yoe_exceeds_career_span  - claims N years of experience but the earliest
                                career entry is far more recent
  2. expert_skills_never_used - multiple "expert" proficiencies with <=6
                                months of actual use
  3. skill_duration_impossible- multiple skills allegedly used for longer
                                than the candidate's entire career
  4. summary_contradicts_yoe  - the free-text summary states a different
                                number of years than the structured field
"""
import re
from datetime import date

GRACE_YEARS = 1.5          # allowance for pre-listed-history experience
SKILL_GRACE_MONTHS = 24    # allowance for skills learned before first job
EXPERT_MIN_MONTHS = 6
MIN_VIOLATING_SKILLS = 2   # one odd skill is noise; two+ is a fabrication
YOE_TEXT_TOLERANCE = 1.5   # summary text vs structured years-of-experience


def _parse_date(s):
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


def career_span_years(candidate, today):
    starts = [j["start_date"] for j in candidate["career_history"]]
    first = _parse_date(min(starts))
    return (today - first).days / 365.25


def consistency_violations(candidate, today):
    """Returns a list of violation tags; empty list = profile is internally
    consistent."""
    violations = []
    span = career_span_years(candidate, today)
    yoe = candidate["profile"]["years_of_experience"]

    if yoe > span + GRACE_YEARS:
        violations.append("yoe_exceeds_career_span")

    expert_unused = sum(
        1 for s in candidate["skills"]
        if s["proficiency"] == "expert"
        and s.get("duration_months", EXPERT_MIN_MONTHS + 1) <= EXPERT_MIN_MONTHS
    )
    if expert_unused >= MIN_VIOLATING_SKILLS:
        violations.append("expert_skills_never_used")

    impossible_durations = sum(
        1 for s in candidate["skills"]
        if s.get("duration_months", 0) > span * 12 + SKILL_GRACE_MONTHS
    )
    if impossible_durations >= MIN_VIOLATING_SKILLS:
        violations.append("skill_duration_impossible")

    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years of experience",
                  candidate["profile"]["summary"])
    if m and abs(float(m.group(1)) - yoe) > YOE_TEXT_TOLERANCE:
        violations.append("summary_contradicts_yoe")

    return violations
