"""Versioned outcome questions and deterministic missing-evidence-aware aggregation."""

from collections import defaultdict

VERSION = "outcomes-v1"
QUESTIONS = [
    (
        "claim_supported",
        "validation",
        "Does a later independent primary study explicitly support the paper's central claim?",
    ),
    (
        "independent_reproduction",
        "validation",
        "Has an independent team reproduced the central result, construction or proof?",
    ),
    (
        "stronger_test",
        "validation",
        "Did the central claim survive a larger, stricter or independently collected follow-up test?",
    ),
    (
        "central_claim_retained",
        "validation",
        "Does a later authoritative assessment explicitly retain the central claim rather than reject it?",
    ),
    (
        "limitations_survived",
        "validation",
        "Did later work show the central contribution remains useful after explicitly testing its stated limitations?",
    ),
    (
        "independent_extension",
        "uptake",
        "Did an independent group publish a substantive extension of this specific contribution?",
    ),
    (
        "multiple_groups",
        "uptake",
        "Did at least two independent groups substantively build on this specific contribution?",
    ),
    (
        "method_reused",
        "uptake",
        "Did an independent study actually use the proposed method, result or framework rather than merely cite it?",
    ),
    (
        "new_research_direction",
        "uptake",
        "Does a later review explicitly identify this contribution as enabling a research direction?",
    ),
    (
        "independent_comparison",
        "uptake",
        "Was this contribution used as an actual baseline, benchmark or theoretical reference construction in independent work?",
    ),
    (
        "external_utility",
        "utility",
        "Was a concrete scientific or practical benefit from applying the contribution demonstrated independently?",
    ),
    (
        "beyond_original_setting",
        "utility",
        "Was the contribution successfully used outside the original evaluation setting or problem instance?",
    ),
    (
        "enabled_new_result",
        "utility",
        "Did the contribution enable a new result, measurement, theorem or capability in later work?",
    ),
    (
        "operational_adoption",
        "utility",
        "Where an operational application is appropriate, was the contribution actually used in a product, workflow, standard or practice?",
    ),
    (
        "advantage_retained",
        "utility",
        "Did an independent comparison find a claimed advantage holds under fair evaluation?",
    ),
    (
        "late_use",
        "durability",
        "Is there evidence of substantive use at least three years after publication?",
    ),
    (
        "late_positive_assessment",
        "durability",
        "Is there a positive independent assessment of the central contribution at least three years after publication?",
    ),
    (
        "multiple_followup_years",
        "durability",
        "Is substantive independent follow-up documented in at least two distinct later calendar years?",
    ),
    (
        "synthesis_inclusion",
        "durability",
        "Does an independent systematic review, textbook, standard or research synthesis adopt the contribution as valid or useful?",
    ),
    (
        "continued_relevance",
        "durability",
        "Does later evidence explicitly show continued usefulness despite newer alternatives?",
    ),
]
IDS = [q[0] for q in QUESTIONS]
DIMENSIONS = ["validation", "uptake", "utility", "durability"]


def aggregate(answers):
    """Known-answer index + worst/best missingness bounds; never unknown=negative."""
    if {a["id"] for a in answers} != set(IDS) or len(answers) != len(IDS):
        raise ValueError("Exactly one answer per rubric question is required")
    groups = defaultdict(list)
    mapping = {q[0]: q[1] for q in QUESTIONS}
    for a in answers:
        if a["answer"] not in ("yes", "no", "unknown", "not_applicable"):
            raise ValueError("Invalid answer state")
        if a["answer"] in ("yes", "no") and not a["evidence"]:
            raise ValueError("Binary answers require evidence")
        groups[mapping[a["id"]]].append(a)
    dimensions = {}
    for name in DIMENSIONS:
        rows = groups[name]
        applicable = [a for a in rows if a["answer"] != "not_applicable"]
        known = [a for a in applicable if a["answer"] in ("yes", "no")]
        yes = sum(a["answer"] == "yes" for a in known)
        n = len(applicable)
        k = len(known)
        dimensions[name] = {
            "yes": yes,
            "no": k - yes,
            "unknown": n - k,
            "not_applicable": len(rows) - n,
            "coverage": k / n if n else None,
            "known_index": yes / k if k else None,
            "lower_bound": yes / n if n else None,
            "upper_bound": (yes + n - k) / n if n else None,
        }
    active = [v for v in dimensions.values() if v["coverage"] is not None]
    known_total = sum(v["yes"] + v["no"] for v in active)
    enough = len(active) >= 3 and all(v["coverage"] >= 0.6 for v in active) and known_total >= 10
    return {
        "dimensions": dimensions,
        "known_answers": known_total,
        "coverage": sum(v["coverage"] for v in active) / len(active) if active else 0,
        "score": sum(v["known_index"] for v in active) / len(active) if enough else None,
        "lower_bound": sum(v["lower_bound"] for v in active) / len(active) if active else None,
        "upper_bound": sum(v["upper_bound"] for v in active) / len(active) if active else None,
        "score_kind": "evidence-backed rubric index; not a probability",
        "eligible_for_aggregate": enough,
    }
