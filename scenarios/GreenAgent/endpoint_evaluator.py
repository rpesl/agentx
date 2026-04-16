import re

"""
This module provides functions to evaluate the performance of an endpoint retrieval system by computing precision, recall, and F1 score. 
It also includes utilities for normalizing and matching API endpoints.
"""

def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(round(f1, 2))

def normalize_endpoint(endpoint: str) -> str:
    match = re.match(r"([A-Z]+)\s+(.+)", endpoint.strip())
    if not match:
        return endpoint.strip()

    method, path = match.groups()
    path = path.strip()

    path = re.sub(r"^/api+/", "/", path)
    path = re.sub(r"^/v[0-9]+/", "/", path)

    return f"{method} {path}"


def normalize_expected_endpoint(endpoint: str) -> re.Pattern:
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
    matches = set()
    for r in retrieved:
        for p in expected_patterns:
            if p.match(r):
                matches.add(r)
                break
    return matches
