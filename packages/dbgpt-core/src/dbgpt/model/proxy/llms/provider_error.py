"""Typed, sanitized errors returned by upstream model providers."""

from dataclasses import dataclass
from typing import Literal, Optional

from dbgpt.core import ModelOutput
from dbgpt.core.schema.api import ErrorCode

UPSTREAM_ERROR_CONTEXT_KEY = "upstream_error"
UpstreamErrorKind = Literal["rate_limit", "upstream_error"]


@dataclass(frozen=True)
class UpstreamProviderError:
    """Machine-readable provider failure without the provider response body."""

    kind: UpstreamErrorKind
    status_code: Optional[int]


def model_output_from_provider_error(error: Exception) -> ModelOutput:
    """Convert an SDK error to a sanitized, typed ``ModelOutput``."""
    status_code = _provider_status_code(error)
    if status_code == 429:
        provider_error = UpstreamProviderError("rate_limit", status_code)
        error_code = ErrorCode.RATE_LIMIT.value
        message = "Upstream model provider rate limited the request."
    else:
        provider_error = UpstreamProviderError("upstream_error", status_code)
        error_code = ErrorCode.INTERNAL_ERROR.value
        message = "Upstream model provider request failed."
    return ModelOutput(
        text=message,
        error_code=error_code,
        model_context={
            UPSTREAM_ERROR_CONTEXT_KEY: {
                "kind": provider_error.kind,
                "status_code": provider_error.status_code,
            }
        },
    )


def upstream_provider_error_from_output(
    output: ModelOutput,
) -> UpstreamProviderError:
    """Read validated provider error metadata, with an error-code fallback."""
    context = output.model_context or {}
    raw_error = context.get(UPSTREAM_ERROR_CONTEXT_KEY)
    if isinstance(raw_error, dict):
        raw_kind = raw_error.get("kind")
        raw_status = raw_error.get("status_code")
        if raw_kind in ("rate_limit", "upstream_error"):
            status_code = _valid_http_status(raw_status)
            return UpstreamProviderError(raw_kind, status_code)
    if output.error_code == ErrorCode.RATE_LIMIT.value:
        return UpstreamProviderError("rate_limit", 429)
    return UpstreamProviderError("upstream_error", None)


def _provider_status_code(error: Exception) -> Optional[int]:
    status_code = _valid_http_status(getattr(error, "status_code", None))
    if status_code is not None:
        return status_code
    response = getattr(error, "response", None)
    return _valid_http_status(getattr(response, "status_code", None))


def _valid_http_status(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 400 <= value <= 599:
        return value
    return None
