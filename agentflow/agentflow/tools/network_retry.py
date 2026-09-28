"""Shared limits and backoff for transient search-tool network failures."""

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


MAX_NETWORK_RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 30.0
RETRYABLE_HTTP_STATUSES = frozenset({403, 412, 422, 429, 500, 502, 503, 504})


def retry_wait_seconds(attempt, *, status_code=None, retry_after=None):
    """Use fixed 1/2/4-second waits except for rate-limited HTTP 429."""
    base_wait = float(2 ** attempt)
    if status_code != 429:
        return base_wait

    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError, AttributeError):
                delay = None
        if delay is not None and math.isfinite(delay):
            return max(0.0, delay) + 0.5

    return base_wait + random.uniform(0.0, 0.5)
