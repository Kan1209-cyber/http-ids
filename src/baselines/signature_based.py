import sys, os, re
sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from urllib.parse import unquote

# A representative subset of real OWASP CRS-style signatures.
# Each is a known, published pattern for detecting a specific attack class.
SIGNATURES = {
    "sqli": [
        r"(\bunion\b.{1,100}\bselect\b)",
        r"(\bselect\b.{1,100}\bfrom\b)",
        r"(\bdrop\b\s+\btable\b)",
        r"(\binsert\b\s+\binto\b)",
        r"(\bor\b\s+\d+\s*=\s*\d+)",
        r"(\band\b\s+\d+\s*=\s*\d+)",
        r"('\s*or\s*')",
        r"(--\s*$)",
        r"(;\s*drop\b)",
    ],
    "xss": [
        r"(<script\b)",
        r"(javascript\s*:)",
        r"(on(error|load|click|mouseover)\s*=)",
        r"(<img\b[^>]*\bonerror\b)",
        r"(alert\s*\()",
        r"(<iframe\b)",
    ],
    "path_traversal": [
        r"(\.\./|\.\.\\)",
        r"(/etc/passwd)",
        r"(boot\.ini)",
        r"(\\windows\\system32)",
    ],
    "command_injection": [
        r"(;\s*(ls|cat|whoami|wget|curl)\b)",
        r"(\|\s*(ls|cat|whoami)\b)",
        r"(`.*`)",
    ],
}

_compiled = {
    category: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for category, patterns in SIGNATURES.items()
}


def run_signature_based(request: dict) -> dict:
    """
    Baseline: classic signature/pattern matching, like ModSecurity + OWASP CRS.
    No structural check, no learned model — just regex matching against
    known attack patterns. This is what pre-ML-era WAFs do.
    """
    query = request.get("query") or ""
    body = request.get("body") or ""
    path = request.get("path") or ""

    content = unquote(query + " " + body + " " + path)

    matched_categories = []
    for category, patterns in _compiled.items():
        for pattern in patterns:
            if pattern.search(content):
                matched_categories.append(category)
                break  # one match per category is enough to flag it

    verdict = "MALICIOUS" if matched_categories else "ALLOWED"

    return {
        "verdict": verdict,
        "matched_categories": matched_categories,
    }