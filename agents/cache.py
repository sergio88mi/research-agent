"""
On-disk cache for API responses.

Why: the proposal uses a free-tier model API (section 3) and free scholarly
APIs, and their quotas are small, so a request that has been answered once is
not sent again.
Caching also makes runs reproducible - the same query returns the same
records - which is what lets the functional tests compare like with like.

Why a file is written under a temporary name and then renamed: a run can be
stopped with Ctrl-C at any moment. A cache file cut off half-way would make
every later run of the same request fail when it is read back. The rename is
atomic, so a file is either complete or absent; and a file that does not
parse is treated as "not asked before" rather than as an error (remediation
#20).
"""
import hashlib
import json
import os
from typing import Any, Optional

import config


def _key(namespace: str, payload: str) -> str:
    """Path of the cache file for one request: the service name plus a hash of what was asked."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return os.path.join(config.CACHE_DIR, f"{namespace}_{digest}.json")


def get(namespace: str, payload: str) -> Optional[Any]:
    """The stored answer to this request, or None if it has not been asked before."""
    path = _key(namespace, payload)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except ValueError:          # damaged file: ask again and overwrite it
                return None
    return None


def put(namespace: str, payload: str, value: Any) -> None:
    """Store the answer to this request (creates the cache directory on first use)."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = _key(namespace, payload)
    partial = f"{path}.{os.getpid()}.part"
    try:
        with open(partial, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):     # the write was interrupted: leave nothing half-made behind
            os.remove(partial)
