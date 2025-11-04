
"""Minimal handshake / integrity helpers.

This module implements a lightweight integrity check for Python objects sent
over ZMQ between the coordinator and workers. It computes a SHA-256 digest of
the pickled message (excluding the digest field itself) and attaches the hex
digest to the message under the key "hash_hex".

This is intentionally minimal (no authenticity or confidentiality). It is
useful as a first step to detect accidental or malicious modification of
messages in transit.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from typing import Any, Dict
import numpy as np
import torch
import hmac
import binascii


def _canonicalize(obj: Any) -> Any:
	"""Recursively transform the object into a structure made only of
	built-in Python types (dict, list, str, int, float, bool, None).

	This ensures deterministic hashing even when the message contains
	numpy arrays or torch tensors which may pickle slightly differently
	between processes.
	"""
	# Primitives
	if obj is None or isinstance(obj, (bool, int, str)):
		return obj
	if isinstance(obj, float):
		# Normalize floats to Python float
		return float(obj)

	# Numpy scalar
	if isinstance(obj, (np.integer, np.floating, np.bool_)):
		return obj.item()

	# Numpy arrays
	if isinstance(obj, np.ndarray):
		return obj.tolist()

	# Torch tensors
	if isinstance(obj, torch.Tensor):
		return obj.detach().cpu().numpy().tolist()

	# Bytes -> hex
	if isinstance(obj, (bytes, bytearray)):
		return bytes(obj).hex()

	# Dict-like
	if isinstance(obj, dict):
		return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}

	# Lists / tuples / sets
	if isinstance(obj, (list, tuple, set)):
		return [_canonicalize(v) for v in obj]

	# Fallback: try to convert via __dict__ or string representation
	if hasattr(obj, "__dict__"):
		return _canonicalize(vars(obj))

	return str(obj)


def _canonical_pickle(obj: Any) -> bytes:
	"""Return a deterministic bytes representation for hashing.

	We first canonicalize the object into basic Python types and then
	pickle with a fixed protocol.
	"""
	canon = _canonicalize(obj)
	return pickle.dumps(canon, protocol=4)


def make_payload_bytes(obj: Any) -> bytes:
	"""Serialize the original object to bytes for transport.

	We use pickle for the payload so the receiver can reconstruct the
	original types (numpy arrays, tensors, etc.).
	"""
	return pickle.dumps(obj, protocol=4)


def compute_hmac_for_obj(obj: Any, key: bytes) -> str:
	"""Compute HMAC-SHA256 hex for an object using its canonical pickle
	representation.
	"""
	canonical = _canonical_pickle(obj)
	mac = hmac.new(key, canonical, digestmod=hashlib.sha256)
	return mac.hexdigest()


def verify_payload(payload_bytes: bytes, recv_hmac_hex: str, key: bytes):
	"""Given raw payload bytes and a received HMAC hex string, unpickle the
	payload and verify the HMAC computed over the canonical form. Returns
	(obj, ok).
	"""
	try:
		obj = pickle.loads(payload_bytes)
	except Exception as e:
		return None, False

	expected = compute_hmac_for_obj(obj, key)
	# Use constant-time compare
	try:
		ok = hmac.compare_digest(expected, recv_hmac_hex)
	except Exception:
		ok = False
	return obj, ok


def compute_hash_for_message(msg: Dict[str, Any]) -> str:
	"""Compute SHA-256 hex digest for a message dict excluding any existing
	'hash_hex' field.
	"""
	# Make a shallow copy and remove the hash field if present.
	m = dict(msg)
	m.pop("hash_hex", None)
	b = _canonical_pickle(m)
	return hashlib.sha256(b).hexdigest()


def attach_hash(msg: Dict[str, Any]) -> None:
	"""Mutate msg to include a 'hash_hex' field computed over the msg's
	contents (excluding 'hash_hex' itself).
	"""
	msg["hash_hex"] = compute_hash_for_message(msg)


def verify_hash(msg: Dict[str, Any]) -> bool:
    """Verify that the 'hash_hex' in msg matches the digest computed over
    the rest of the message. Returns True if matching, False otherwise.
    """
    if "hash_hex" not in msg:
        return False
    expected = msg["hash_hex"]
    actual = compute_hash_for_message(msg)
    # Use constant-time comparison for safety
    return hmac.compare_digest(expected, actual)


