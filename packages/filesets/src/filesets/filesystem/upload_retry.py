# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry decisions for streamed local-file uploads.

``put_file`` streams the local file as a one-shot body, so once the body is on
the wire the typed client will not send it again. The filesystem can reopen the
file, so it retries those failures itself, under the client's own
:class:`RetryPolicy`. A client built with ``RetryPolicy(max_retries=0)`` or no
policy is never retried here either.
"""

from __future__ import annotations

import email.utils
import time
from datetime import timezone

import httpx
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client.types import RetryPolicy

# Failures raised before httpx reads any of the body. The typed client already
# retries these itself with the untouched body, so retrying again here would
# multiply its attempts.
_PRE_BODY_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)

_MAX_RETRY_AFTER_SECONDS = 60.0


def upload_retry_delay(exc: Exception, attempt: int, policy: RetryPolicy | None) -> float | None:
    """Seconds to wait before re-sending the upload, or ``None`` to raise ``exc``.

    ``attempt`` counts from 0 for the first failed upload.
    """
    if policy is None or attempt >= policy.max_retries:
        return None
    backoff = policy.backoff_base * (2**attempt)

    if isinstance(exc, NemoTransportError):
        return None if isinstance(exc.error, _PRE_BODY_TRANSPORT_ERRORS) else backoff
    if not isinstance(exc, NemoHTTPError):
        return None

    response = exc.http_response
    decision = response.headers.get("x-should-retry") if policy.respect_retry_decision_headers else None
    if decision == "false":
        return None
    if decision != "true":
        retryable = exc.status_code in policy.retryable_status_codes or (
            policy.retry_all_server_errors and exc.status_code >= 500
        )
        if not retryable:
            return None
    if policy.respect_retry_after_headers:
        return _retry_after(response) or backoff
    return backoff


def _retry_after(response: httpx.Response) -> float | None:
    """Server-requested delay from ``retry-after-ms`` or ``Retry-After``, if reasonable."""
    headers = response.headers
    try:
        delay = float(headers["retry-after-ms"]) / 1000
    except (KeyError, ValueError):
        value = headers.get("retry-after")
        if value is None:
            return None
        try:
            delay = float(value)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delay = when.timestamp() - time.time()
    return delay if 0 < delay <= _MAX_RETRY_AFTER_SECONDS else None
