from typing import List
from server.models import IssueReport

MATCH_WEIGHTS = {
    "issue_type": 0.2,
    "severity": 0.3,
    "line_proximity": 0.3,   # within ±3 lines counts as match
    "description_keywords": 0.2
}


def extract_keywords(text: str) -> set:
    stopwords = {
        "a", "an", "the", "is", "in", "to", "of", "and", "or", "it",
        "be", "this", "that", "will", "can", "on", "at", "by", "for",
        "with", "as", "from", "are", "was", "not", "if", "its", "may"
    }
    return set(w.lower().strip(".,;:\"'()[]{}") for w in text.split() if w.lower() not in stopwords)


def match_issue(reported: IssueReport, ground_truth: IssueReport) -> float:
    score = 0.0
    if reported.issue_type == ground_truth.issue_type:
        score += MATCH_WEIGHTS["issue_type"]
    if reported.severity == ground_truth.severity:
        score += MATCH_WEIGHTS["severity"]
    if reported.line_number is not None and ground_truth.line_number is not None:
        if abs(reported.line_number - ground_truth.line_number) <= 3:
            score += MATCH_WEIGHTS["line_proximity"]
    reported_kw = extract_keywords(reported.description)
    truth_kw = extract_keywords(ground_truth.description)
    if truth_kw:
        overlap = len(reported_kw & truth_kw) / len(truth_kw)
        score += MATCH_WEIGHTS["description_keywords"] * overlap
    return score


def grade(reported_issues: List[IssueReport], ground_truth_issues: List[IssueReport]) -> dict:
    if not ground_truth_issues:
        return {
            "score": 0.999,
            "recall": 0.999,
            "precision": 0.999,
            "f1": 0.999,
            "matched": 0,
            "false_positives": len(reported_issues)
        }

    matched_gt = set()
    true_positives = 0

    for reported in reported_issues:
        best_score = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truth_issues):
            if idx in matched_gt:
                continue
            s = match_issue(reported, gt)
            if s > best_score:
                best_score = s
                best_idx = idx
        if best_score >= 0.5 and best_idx >= 0:
            matched_gt.add(best_idx)
            true_positives += 1

    recall = true_positives / len(ground_truth_issues)
    precision = true_positives / len(reported_issues) if reported_issues else 0.0
    f1 = 2 * recall * precision / (recall + precision + 1e-9)
    false_positives = len(reported_issues) - true_positives
    fp_penalty = min(0.2, false_positives * 0.05)

    final_score = max(0.001, min(0.999, f1 - fp_penalty))

    return {
        "score": round(final_score, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "matched": true_positives,
        "false_positives": false_positives
    }