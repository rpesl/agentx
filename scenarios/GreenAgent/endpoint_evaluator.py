import re
from math import comb

"""
This module provides functions to evaluate the performance of an endpoint retrieval system by computing precision, recall, F1 score, and Pass@k. 
It also includes utilities for normalizing and matching API endpoints.
"""

def compute_f1(precision: float, recall: float) -> float:
    """Compute the F1 score given precision and recall."""
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(round(f1, 2))

def compute_pass_at_k(num_correct: int, num_total: int, k: int) -> float:
    """Compute Pass@k metric for endpoint retrieval."""
    if k > num_total:
        k = num_total

    if num_total == 0 or num_correct == 0:
        return 0.0

    if num_correct == num_total:
        return 1.0

    try:
        pass_at_k = 1.0 - (comb(num_total - num_correct, k) / comb(num_total, k))
        return float(round(pass_at_k, 2))
    except (ValueError, ZeroDivisionError):
        return 0.0

def normalize_endpoint(endpoint: str) -> str:
    """Normalize an API endpoint by removing versioning and common prefixes."""
    match = re.match(r"([A-Z]+)\s+(.+)", endpoint.strip())
    if not match:
        return endpoint.strip()

    method, path = match.groups()
    path = path.strip()

    path = re.sub(r"^/api+/", "/", path)
    path = re.sub(r"^/v[0-9]+/", "/", path)

    return f"{method} {path}"

def normalize_expected_endpoint(endpoint: str) -> re.Pattern:
    """Convert an expected endpoint pattern into a regex pattern for matching retrieved endpoints."""
    endpoint = endpoint.strip()
    match = re.match(r"([A-Z]+)\s+(.+)", endpoint)
    if not match:
        raise ValueError(f"Invalid endpoint: {endpoint}")

    method, path = match.groups()
    path = path.strip()

    path_regex = re.sub(r"\{[^}]+}", r"[^/]*", path)
    path_regex = re.sub(r"<[^>]+>", r"[^/]*", path_regex)

    full_regex = f"^{method} {path_regex}$"
    return re.compile(full_regex)

def match_retrieved_to_expected(retrieved: set[str], expected_patterns: list[re.Pattern]):
    """Match retrieved endpoints against expected endpoint patterns and return the set of matches."""
    matches = set()
    for r in retrieved:
        for p in expected_patterns:
            if p.match(r):
                matches.add(r)
                break
    return matches
