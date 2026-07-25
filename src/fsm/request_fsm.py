# src/fsm/request_fsm.py

from enum import Enum, auto


class State(Enum):
    START = auto()
    REQUEST_LINE = auto()
    HEADERS = auto()
    BODY = auto()
    COMPLETE = auto()
    REJECTED = auto()


class RequestFSM:
    """
    Validates the structural legality of an HTTP request.
    Does NOT inspect content/payloads — only shape and header consistency.
    """

    def __init__(self):
        self.state = State.START
        self.reject_reason = None

    def _reject(self, reason: str):
        self.state = State.REJECTED
        self.reject_reason = reason
        return self.state

    def process(self, request: dict) -> State:
        """
        request: {
            "method": str,
            "path": str,
            "http_version": str,
            "headers": dict[str, str],
            "body": str | None,
        }
        """
        self.state = State.START

        # START -> REQUEST_LINE
        if not self._valid_request_line(request):
            return self._reject("malformed request line")
        self.state = State.REQUEST_LINE

        # REQUEST_LINE -> HEADERS
        headers = request.get("headers", {})
        if not self._valid_headers(headers):
            return self.state  # _valid_headers sets REJECTED itself
        self.state = State.HEADERS

        # HEADERS -> BODY (optional) -> COMPLETE
        if request.get("body") is not None:
            if not self._valid_body_framing(headers, request["body"]):
                return self.state
            self.state = State.BODY

        self.state = State.COMPLETE
        return self.state

    def _valid_request_line(self, request: dict) -> bool:
        method = request.get("method")
        path = request.get("path")
        version = request.get("http_version")
        valid_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
        if method not in valid_methods:
            return False
        if not path or not path.startswith("/"):
            return False
        if version not in {"HTTP/1.0", "HTTP/1.1"}:
            return False
        return True

    def _valid_headers(self, headers: dict) -> bool:
        # Case-insensitive lookup
        norm = {k.lower(): v for k, v in headers.items()}

        has_cl = "content-length" in norm
        has_te = "transfer-encoding" in norm

        # THE smuggling check
        if has_cl and has_te:
            self._reject("conflicting Content-Length and Transfer-Encoding")
            return False

        if has_cl:
            try:
                cl = int(norm["content-length"])
                if cl < 0:
                    self._reject("negative content-length")
                    return False
            except ValueError:
                self._reject("non-integer content-length")
                return False

        if has_te and norm["transfer-encoding"].lower() not in {"chunked", "identity"}:
            self._reject("unsupported transfer-encoding")
            return False

        return True

    def _valid_body_framing(self, headers: dict, body: str) -> bool:
        norm = {k.lower(): v for k, v in headers.items()}
        if "content-length" in norm:
            declared = int(norm["content-length"])
            if declared != len(body):
                self._reject(f"content-length mismatch: declared {declared}, actual {len(body)}")
                return False
        return True