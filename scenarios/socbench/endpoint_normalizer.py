import re

def normalize_endpoint(endpoint: str) -> str:
    match = re.match(r"([A-Z]+)\s+(.+)", endpoint.strip())
    if not match:
        return endpoint.strip()

    method, path = match.groups()
    path = path.strip()

    path = re.sub(r"^/v[0-9]+/", "/", path)

    segments = path.split("/")
    normalized_segments = []

    for seg in segments:
        if not seg:
            continue

        if (re.search(r"\d", seg)
                or re.fullmatch(r"\{.*?}", seg)
                or re.fullmatch(r"<.*?>", seg)):
                normalized_segments.append("{id}")
        else:
            normalized_segments.append(seg)

    normalized_path = "/" + "/".join(normalized_segments)
    return f"{method} {normalized_path}"