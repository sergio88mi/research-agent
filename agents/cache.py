"""
On-disk cache for API responses.

Why: free-tier quotas are small and the proposal commits to not exhausting them
during development. Caching also makes runs reproducible - the same query
returns the same records - which is what lets the functional tests compare
like with like.
"""
import hashlib
import json
import os
from typing import Any, Optional

import config


def _key(namespace: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return os.path.join(config.CACHE_DIR, f"{namespace}_{digest}.json")


def get(namespace: str, payload: str) -> Optional[Any]:
    path = _key(namespace, payload)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def put(namespace: str, payload: str, value: Any) -> None:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(_key(namespace, payload), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
