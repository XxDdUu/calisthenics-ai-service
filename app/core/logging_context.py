"""
Logging context tracking using standard library contextvars.
Allows propagating request_id across async and synchronous call stacks.
"""
import contextvars
from typing import Optional

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Retrieve the current request ID from context, or '-' if not set."""
    return _request_id_ctx.get()


def set_request_id(request_id: Optional[str]) -> str:
    """Set the current request ID in context."""
    rid = request_id if request_id else "-"
    _request_id_ctx.set(rid)
    return rid
