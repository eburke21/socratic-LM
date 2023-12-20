"""Retry utility with exponential backoff for OpenAI API calls."""

import time
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# Retryable error types — transient failures that may succeed on retry
RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# Default retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds — doubles on each retry (1s, 2s, 4s)


def retry_api_call(fn, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """Call `fn(*args, **kwargs)` with exponential backoff on retryable OpenAI errors.

    Returns the result of fn on success.
    Raises the last error if all retries are exhausted.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < max_retries:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  [retry] {type(e).__name__} — retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  [retry] {type(e).__name__} — all {max_retries} retries exhausted")
    raise last_error
