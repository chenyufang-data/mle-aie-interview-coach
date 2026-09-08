"""Tiny JSON request/response helpers shared by the handler and dev routes."""

import json

# Bytes. Routes that carry a file (resume upload, answer audio) raise it per
# route in coach/http.py; everything else is text and fits in 1 MB.
DEFAULT_BODY_LIMIT = 1_000_000


class PayloadTooLarge(ValueError):
    """Request body above the route's limit (answered with HTTP 413)."""


def json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler, limit=DEFAULT_BODY_LIMIT):
    """Parse the JSON body, refusing anything above `limit` bytes BEFORE it
    is read, so an anonymous client cannot make the server hold an
    arbitrary upload in memory. Raises PayloadTooLarge (413) or ValueError
    (400) for the handler to answer."""
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("Content-Length is not a number")
    if length < 0:
        raise ValueError("Content-Length is negative")
    if length == 0:
        return {}
    if length > limit:
        raise PayloadTooLarge(
            f"request body of {length:,} bytes exceeds this endpoint's "
            f"{limit:,}-byte limit")
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON ({type(exc).__name__})")
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data
