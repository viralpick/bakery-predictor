"""Shared HTTP helpers for data.go.kr APIs.

Notes:
- All data.go.kr endpoints expect `ServiceKey` as a query param. httpx
  URL-encodes the value, so pass the *decoded* key (the one with `/` and
  `==`). The portal labels it "일반 인증키 (Decoding)".
- These APIs return small payloads (≤ a few KB per page) but rate-limit
  silently — we throttle modestly between calls.
"""

from __future__ import annotations

import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_THROTTLE_SECONDS = 0.1


class ApiError(RuntimeError):
    """Raised when a data.go.kr endpoint returns a non-OK resultCode."""


def get_xml(url: str, params: dict[str, Any], *, retries: int = DEFAULT_RETRIES) -> ET.Element:
    """GET an endpoint and parse its XML response, validating resultCode."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            _check_result(root, url=url)
            time.sleep(DEFAULT_THROTTLE_SECONDS)
            return root
        except (httpx.HTTPError, ET.ParseError) as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} retries: {url}") from last_exc


def _check_result(root: ET.Element, *, url: str) -> None:
    code = root.findtext(".//resultCode")
    if code is None:
        # Some APIs nest header differently; fall back to header/resultCode
        code = root.findtext(".//header/resultCode")
    if code is not None and code != "00":
        msg = root.findtext(".//resultMsg") or root.findtext(".//header/resultMsg") or "(no msg)"
        raise ApiError(f"{url} returned resultCode={code}: {msg}")


def iter_items(root: ET.Element) -> list[dict[str, str | None]]:
    """Flatten <item> children to list-of-dict. Empty months return []."""
    items = root.findall(".//item")
    return [{child.tag: child.text for child in item} for item in items]
