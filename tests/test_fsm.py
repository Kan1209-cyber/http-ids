from src.fsm.request_fsm import RequestFSM, State

def test_valid_get():
    fsm = RequestFSM()
    req = {"method": "GET", "path": "/login", "http_version": "HTTP/1.1", "headers": {}}
    assert fsm.process(req) == State.COMPLETE

def test_smuggling_rejected():
    fsm = RequestFSM()
    req = {
        "method": "POST", "path": "/", "http_version": "HTTP/1.1",
        "headers": {"Content-Length": "44", "Transfer-Encoding": "chunked"},
    }
    assert fsm.process(req) == State.REJECTED
    assert "conflicting" in fsm.reject_reason