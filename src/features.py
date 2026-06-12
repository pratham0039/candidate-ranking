"""
Candidate feature extraction and scoring components.

Turns a candidate profile plus a parsed JD spec into interpretable scoring
factors. The final ranking score is a product of these factors, so each one
is scaled to a sensible range and can be inspected on its own. This is what
lets the system generate honest, specific reasoning for every ranked
candidate: the reasoning is read off the factors, not generated separately.

Factors:
  evidence_score   : how strongly the candidate's career history demonstrates
                     the JD's must-have and nice-to-have competencies. Uses
                     career descriptions, titles and corroborated skills, not
                     the self-reported skills list alone, because skill lists
                     are cheap to inflate.
  yoe_score        : trapezoid over the JD's experience band. Full score
                     inside the ideal band, partial credit at the edges,
                     smooth decay outside. Soft by design: the JD itself says
                     strong candidates outside the band are considered.
  location_score   : preferred cities, then welcome cities, then rest of the
                     JD's country, then abroad. Relocation willingness
                     softens the penalty; no visa sponsorship hardens it.
  availability     : platform engagement. Exponential recency decay on last
                     activity, recruiter response rate, open-to-work flag,
                     interview completion, notice period step function.
  penalties        : multiplicative dampeners for career patterns the JD
                     explicitly rejects: services-firms-only history, very
                     short median tenure with title escalation, vision or
                     speech specialization with no NLP/IR exposure.
"""
import math
import re
from datetime import date

from jd_parser import (CONCEPT_LEXICONS, CONSULTING_FIRMS,
                       CV_SPEECH_MARKERS, NLP_IR_MARKERS)


def _parse_date(s):
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


def candidate_evidence_text(candidate):
    """The text we trust: career descriptions, titles, headline, summary.
    Listed skills are appended only when corroborated by a platform
    assessment score or by meaningful usage duration."""
    parts = [candidate["profile"]["headline"], candidate["profile"]["summary"]]
    for job in candidate["career_history"]:
        parts.append(job["title"])
        parts.append(job["description"])
    assessments = candidate["redrob_signals"].get("skill_assessment_scores", {})
    for s in candidate["skills"]:
        corroborated = (
            assessments.get(s["name"], 0) >= 60
            or s.get("duration_months", 0) >= 24
        )
        if corroborated:
            parts.append(s["name"])
    return " ".join(parts).lower()


def evidence_score(candidate, spec, evidence_text=None):
    """Fraction of must-have concepts demonstrated, with a small bonus for
    nice-to-haves. Range roughly 0..1.3."""
    text = evidence_text or candidate_evidence_text(candidate)
    must_hits = 0
    for concept in spec.must_have:
        forms = CONCEPT_LEXICONS.get(concept, [])
        if any(f in text for f in forms):
            must_hits += 1
    nice_hits = 0
    for concept in spec.nice_to_have:
        forms = CONCEPT_LEXICONS.get(concept, [])
        if any(f in text for f in forms):
            nice_hits += 1
    must_frac = must_hits / max(len(spec.must_have), 1)
    nice_frac = nice_hits / max(len(spec.nice_to_have), 1)
    return must_frac + 0.3 * nice_frac


def yoe_score(candidate, spec):
    """Trapezoid: 1.0 inside the ideal band, 0.7 at the stated band edges,
    exponential decay outside the stated band."""
    yoe = candidate["profile"]["years_of_experience"]
    lo, hi = spec.yoe_min, spec.yoe_max
    ilo, ihi = spec.yoe_ideal_min, spec.yoe_ideal_max
    if ilo <= yoe <= ihi:
        return 1.0
    if lo <= yoe < ilo:
        return 0.7 + 0.3 * (yoe - lo) / max(ilo - lo, 1e-6)
    if ihi < yoe <= hi:
        return 0.7 + 0.3 * (hi - yoe) / max(hi - ihi, 1e-6)
    if yoe < lo:
        return 0.7 * math.exp(-(lo - yoe) / 2.0)
    return 0.7 * math.exp(-(yoe - hi) / 3.0)


def location_score(candidate, spec):
    p = candidate["profile"]
    rs = candidate["redrob_signals"]
    loc = p["location"].lower()
    if p["country"] == spec.country:
        if any(c in loc for c in spec.preferred_cities):
            return 1.0
        if any(c in loc for c in spec.welcome_cities):
            return 0.9
        return 0.75 if rs["willing_to_relocate"] else 0.55
    if spec.visa_sponsorship:
        return 0.5
    return 0.35 if rs["willing_to_relocate"] else 0.2


def availability_score(candidate, spec, today, half_life_days=45.0):
    """Engagement signals combined into one multiplier, range roughly 0..1.

    A candidate who never responds and has not logged in for months is not
    hireable regardless of skills, so this multiplies the whole score."""
    rs = candidate["redrob_signals"]

    inactive_days = (today - _parse_date(rs["last_active_date"])).days
    recency = math.exp(-max(inactive_days, 0) * math.log(2) / half_life_days)

    response = rs["recruiter_response_rate"]
    interview = max(rs["interview_completion_rate"], 0.0)
    open_flag = 1.0 if rs["open_to_work_flag"] else 0.0

    notice = rs["notice_period_days"]
    if notice <= spec.notice_pref_days:
        notice_factor = 1.0
    elif notice <= spec.notice_pref_days + spec.notice_buyout_days:
        notice_factor = 0.85
    elif notice <= 90:
        notice_factor = 0.65
    else:
        notice_factor = 0.5

    engagement = (0.35 * recency + 0.30 * response +
                  0.20 * open_flag + 0.15 * interview)
    # floor keeps a perfect-skills candidate visible even when dormant,
    # just heavily downweighted
    return (0.25 + 0.75 * engagement) * notice_factor


# Ownership language: the candidate drove the work. Hedge language: the
# candidate was adjacent to the work or is still aspiring to it. Two
# profiles can mention the same technologies while sitting on opposite
# sides of this line, and recruiters read the difference instantly.
OWNERSHIP_MARKERS = [
    "owned", "led ", "designed", "from scratch", "end-to-end", "end to end",
    "rolled out", "drove", "architected", "shipped", "serving", "queries per",
    "at scale", "migrated", "migration", "built and", "rebuilt", "a/b test",
    "production rollout", "incident", "on-call", "p95", "latency",
]
HEDGE_MARKERS = [
    "lightweight", "lighter weight", "handled by the platform team",
    "pure ml side", "side project", "kaggle", "self-directed",
    "online courses", "dashboard", "analytics side", "building competence",
    "want to grow", "looking to grow", "transitioning toward",
    "self-learner", "played with", "experimented with", "curious about",
    "adjacent ml exposure", "some basic ml", "not the core",
]


def ownership_score(candidate, evidence_text=None):
    """Ratio of ownership language to hedge language across the career
    descriptions and summary. Returns roughly 0.3 (all hedges) to 1.0
    (clear ownership)."""
    text = evidence_text or candidate_evidence_text(candidate)
    own = sum(1 for m in OWNERSHIP_MARKERS if m in text)
    hedge = sum(1 for m in HEDGE_MARKERS if m in text)
    return (1.0 + own) / (1.0 + own + 1.6 * hedge)


def median(values):
    s = sorted(values)
    return s[len(s) // 2] if s else 0


def penalty_factors(candidate, evidence_text=None):
    """Multiplicative dampeners for career patterns the JD rejects.
    Returns (factor, list_of_reasons)."""
    factor = 1.0
    reasons = []
    text = evidence_text or candidate_evidence_text(candidate)
    history = candidate["career_history"]

    companies = {j["company"].lower() for j in history}
    if companies and all(
        any(firm in comp for firm in CONSULTING_FIRMS) for comp in companies
    ):
        factor *= 0.5
        reasons.append("services-firms-only career history")

    # short median tenure with steadily escalating titles
    tenures = [j["duration_months"] for j in history if not j["is_current"]]
    if len(history) >= 3 and tenures and median(tenures) < 18:
        seniority_words = ("senior", "staff", "lead", "principal", "head")
        titles = [j["title"].lower() for j in history]
        escalation = sum(any(w in t for w in seniority_words) for t in titles)
        if escalation >= 2:
            factor *= 0.7
            reasons.append("frequent moves with title escalation")

    # vision or speech specialization without NLP or retrieval exposure
    cv_hits = sum(1 for m in CV_SPEECH_MARKERS if m in text)
    nlp_hits = sum(1 for m in NLP_IR_MARKERS if m in text)
    if cv_hits >= 2 and nlp_hits == 0:
        factor *= 0.6
        reasons.append("vision or speech focus with no NLP or retrieval work")

    return factor, reasons


def engineering_title_gate(candidate):
    """Cheap relevance gate. The role is a hands-on AI engineering position,
    so candidates whose current title and entire career history show no
    software, data or ML work cannot be a fit at any rank."""
    tech_markers = (
        "software", "developer", "devops", "cloud", "frontend", "backend",
        "full stack", "mobile", "qa engineer", "data", "analytics",
        "ml", "machine learning", "nlp", "search", "recommendation",
        "scientist", "programmer", "sde", "swe", "java", ".net",
        "computer vision", "applied",
    )
    titles = [candidate["profile"]["current_title"].lower()] + [
        j["title"].lower() for j in candidate["career_history"]
    ]
    # pad with spaces so "ai" matches as a word ("Head of AI", "AI Lead")
    # without matching inside words like "trainer"
    return any(
        any(m in t for m in tech_markers) or " ai " in f" {t} "
        for t in titles
    )
