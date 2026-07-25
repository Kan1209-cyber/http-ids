import re
import math
from collections import Counter
from urllib.parse import unquote


SQLI_KEYWORDS = ["select", "union", "drop", "insert", "delete", "update", "--", "'", "or 1=1", "and 1=1"]
XSS_KEYWORDS = ["<script", "onerror=", "onload=", "javascript:", "<img", "alert("]
PATH_TRAVERSAL_KEYWORDS = ["../", "..\\", "/etc/passwd", "boot.ini"]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(request: dict) -> dict:
    """
    Takes a parsed request dict (from request_parser) and returns a flat
    dict of numeric/categorical features for the ML model.
    """
    path = request.get("path") or ""
    query = request.get("query") or ""
    body = request.get("body") or ""
    headers = request.get("headers") or {}

    # Combine query + body since both carry attacker-controlled content
    # URL-decode first — attack payloads are often percent-encoded
    # (e.g. "<script" -> "%3Cscript", "../" -> "%2e%2e%2f")
    raw_content = query + " " + body
    content = unquote(raw_content)
    content_lower = content.lower()

    num_params = content.count("=") if content else 0
    special_char_count = sum(content.count(c) for c in ["'", '"', "<", ">", ";", "--", "%"])

    features = {
        "method": request.get("method", ""),
        "path_length": len(path),
        "query_length": len(query),
        "body_length": len(body),
        "content_length": len(content),
        "num_params": num_params,
        "num_headers": len(headers),
        "special_char_count": special_char_count,
        "content_entropy": shannon_entropy(content),
        "sqli_keyword_count": sum(content_lower.count(k) for k in SQLI_KEYWORDS),
        "xss_keyword_count": sum(content_lower.count(k) for k in XSS_KEYWORDS),
        "path_traversal_keyword_count": sum(content_lower.count(k) for k in PATH_TRAVERSAL_KEYWORDS),
        "max_param_length": max((len(p) for p in content.split("&")), default=0),
    }

    return features