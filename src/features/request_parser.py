def parse_requests_from_file(filepath):
    """
    Parses a CSIC 2010 raw traffic file into a list of request dicts:
    {
        "method": str,
        "url": str,          # full original URL as it appeared
        "path": str,         # just the path component (for FSM)
        "query": str,        # query string, if any
        "http_version": str,
        "headers": dict,
        "body": str or None,
    }
    """
    from urllib.parse import urlparse

    requests = []

    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip("\n").rstrip("\r")

        # Skip blank lines between requests
        if not line.strip():
            i += 1
            continue

        # This should be a request line: METHOD URL HTTP/x.x
        parts = line.split(" ")
        if len(parts) != 3:
            i += 1
            continue

        method, url, http_version = parts
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query

        i += 1
        headers = {}

        # Read headers until blank line
        while i < n:
            hline = lines[i].rstrip("\n").rstrip("\r")
            if not hline.strip():
                i += 1
                break
            if ":" in hline:
                key, value = hline.split(":", 1)
                headers[key.strip()] = value.strip()
            i += 1

        # If Content-Length present, the next line is the body
        body = None
        if "Content-Length" in headers or "content-length" in headers:
            if i < n:
                body = lines[i].rstrip("\n").rstrip("\r")
                i += 1

        requests.append({
            "method": method,
            "url": url,
            "path": path,
            "query": query,
            "http_version": http_version,
            "headers": headers,
            "body": body,
        })

    return requests