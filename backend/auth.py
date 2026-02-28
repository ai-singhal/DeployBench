"""API key generation and verification."""

from __future__ import annotations

import hashlib
import secrets



def generate_api_key() -> str:
    return secrets.token_urlsafe(32)



def hash_api_key(api_key: str) -> str:
    if not api_key:
        raise ValueError("api_key cannot be empty")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()



def verify_api_key(candidate: str, expected_hash: str) -> bool:
    if not candidate or not expected_hash:
        return False
    return hash_api_key(candidate) == expected_hash
