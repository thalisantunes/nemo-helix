# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which failed uploads put_file sends again, and after how long.

The resend itself is covered against the real Files app in
``services/core/files/tests/integration/test_fileset_filesystem.py::TestPutFileRetry``.
"""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from filesets.filesystem.upload_retry import upload_retry_delay
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY, RetryPolicy

POLICY = PLATFORM_DEFAULT_RETRY_POLICY
REQUEST = httpx.Request("PUT", "http://test/upload")


def _status(code: int, **headers: str) -> NemoHTTPError:
    return NemoHTTPError(httpx.Response(code, headers=headers, request=REQUEST))


def _transport(error: type[httpx.TransportError]) -> NemoTransportError:
    return NemoTransportError(error("injected", request=REQUEST))


@pytest.mark.parametrize(
    "exc,policy,expected",
    [
        pytest.param(_transport(httpx.ReadTimeout), POLICY, POLICY.backoff_base, id="mid-body-timeout-resent"),
        # The client already retried a refused connection with the untouched body.
        pytest.param(_transport(httpx.ConnectError), POLICY, None, id="connect-error-left-to-client"),
        pytest.param(_status(403), POLICY, None, id="non-retryable-status"),
        # Callers that retry on their own, such as customizer file_io, turn this off.
        pytest.param(_status(503), replace(POLICY, max_retries=0), None, id="caller-opted-out"),
        pytest.param(_status(503), None, None, id="client-without-policy"),
        pytest.param(_status(429, **{"Retry-After": "7"}), POLICY, 7.0, id="retry-after-honored"),
        pytest.param(_status(503, **{"x-should-retry": "false"}), POLICY, None, id="server-declines-retry"),
    ],
)
def test_upload_retry_delay(exc: NemoClientError, policy: RetryPolicy | None, expected: float | None) -> None:
    assert upload_retry_delay(exc, 0, policy) == expected
