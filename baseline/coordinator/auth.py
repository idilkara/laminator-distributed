import os
import json
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from binascii import unhexlify
from binascii import hexlify
from typing import Optional
import time


def sign_message(message: bytes, private_key_file: str) -> bytes:
    """Sign raw bytes with an RSA private key (PKCS#1 v1.5 + SHA256)."""
    key = RSA.import_key(open(private_key_file, "rb").read())
    h = SHA256.new(message)
    signature = pkcs1_15.new(key).sign(h)
    return signature


def verify_signature(message: bytes, signature: bytes, public_key_file: str) -> bool:
    """Verify an RSA signature (PKCS#1 v1.5 + SHA256)."""
    key = RSA.import_key(open(public_key_file, "rb").read())
    h = SHA256.new(message)
    try:
        pkcs1_15.new(key).verify(h, signature)
        return True
    except (ValueError, TypeError):
        return False


def generate_nonce(length: int = 32) -> str:
    """Generate a cryptographically secure nonce as a hex string.

    Args:
        length: number of random bytes to generate (default 32).

    Returns:
        Hex-encoded nonce string.
    """
    return hexlify(os.urandom(length)).decode()


def create_handshake_offer(session_id: str, coordinator_private_key_file: str, coordinator_public_key_file: Optional[str] = None, nonce_len: int = 32) -> dict:
    """Create a coordinator-side handshake offer containing a nonce and signature.

    The offer payload contains the session_id and nonce. The payload is signed
    with the coordinator private key so the worker can verify authenticity.

    Args:
        session_id: Identifier for this session.
        coordinator_private_key_file: Path to the coordinator RSA private key (PEM).
        coordinator_public_key_file: Optional path to coordinator public key file to include in the offer.
        nonce_len: Length in bytes of generated nonce.

    Returns:
        A dict with keys: session_id, nonce, signature (hex), and optional pubkey (PEM string).
    """
    nonce = generate_nonce(nonce_len)
    # Use canonical JSON encoding so signature is deterministic across processes
    payload = json.dumps({"session_id": session_id, "nonce": nonce}, separators=(",", ":"), sort_keys=True).encode()
    signature = sign_message(payload, coordinator_private_key_file)
    offer = {
        "session_id": session_id,
        "nonce": nonce,
        "signature": signature.hex(),
    }
    if coordinator_public_key_file:
        offer["pubkey"] = open(coordinator_public_key_file, "rb").read().decode()
    return offer


def create_handshake_response(offer: dict, worker_id: str, worker_private_key_file: str) -> dict:
    """Create a worker-side handshake response that proves possession of the worker key.

    The response signs the offer's session_id and nonce together with the worker_id.

    Args:
        offer: The handshake offer dict created by `create_handshake_offer`.
        worker_id: Identifier for the worker (string).
        worker_private_key_file: Path to the worker's private key (PEM).

    Returns:
        A dict with keys: session_id, worker_id, nonce, signature (hex).
    """
    payload = json.dumps({
        "session_id": offer["session_id"],
        "nonce": offer["nonce"],
        "worker_id": worker_id,
        "ts": int(time.time()),
    }).encode()
    signature = sign_message(payload, worker_private_key_file)
    return {
        "session_id": offer["session_id"],
        "worker_id": worker_id,
        "nonce": offer["nonce"],
        "signature": signature.hex(),
    }


def verify_handshake_offer(offer: dict, coordinator_public_key_file: str) -> bool:
    """Verify a coordinator handshake offer signature and basic structure.

    Args:
        offer: dict returned by `create_handshake_offer`.
        coordinator_public_key_file: path to coordinator public key file (PEM).

    Returns:
        True if signature is valid and required fields present, False otherwise.
    """
    try:
        session_id = offer["session_id"]
        nonce = offer["nonce"]
        signature = bytes.fromhex(offer["signature"])
    except Exception as e:
        print(f"verify_handshake_offer: malformed offer structure: {e}", flush=True)
        return False
    payload = json.dumps({"session_id": session_id, "nonce": nonce}, separators=(",", ":"), sort_keys=True).encode()
    ok = verify_signature(payload, signature, coordinator_public_key_file)
    if not ok:
        print(f"verify_handshake_offer: signature verification FAILED for session {session_id}", flush=True)
    else:
        print(f"verify_handshake_offer: signature verified for session {session_id}", flush=True)
    return ok


def verify_handshake_response(response: dict, worker_public_key_file: str, expected_session_id: Optional[str] = None, expected_nonce: Optional[str] = None) -> bool:
    """Verify a worker handshake response: signature, session and nonce match.

    Args:
        response: dict returned by `create_handshake_response`.
        worker_public_key_file: path to worker public key file (PEM).
        expected_session_id: optional expected session id to assert equality.
        expected_nonce: optional expected nonce to assert equality.

    Returns:
        True if checks pass, False otherwise.
    """
    try:
        session_id = response["session_id"]
        nonce = response["nonce"]
        signature = bytes.fromhex(response["signature"])
    except Exception as e:
        print(f"verify_handshake_response: malformed response: {e}", flush=True)
        return False
    if expected_session_id is not None and session_id != expected_session_id:
        print(f"verify_handshake_response: session_id mismatch (expected={expected_session_id} got={session_id})", flush=True)
        return False
    if expected_nonce is not None and nonce != expected_nonce:
        print(f"verify_handshake_response: nonce mismatch (expected={expected_nonce} got={nonce})", flush=True)
        return False
    # worker included worker_id and ts when signing payload; verify using the same canonical JSON
    worker_id = response.get("worker_id")
    ts = response.get("ts")
    payload_dict = {"session_id": session_id, "nonce": nonce}
    if worker_id is not None:
        payload_dict["worker_id"] = worker_id
    if ts is not None:
        payload_dict["ts"] = ts
    payload = json.dumps(payload_dict, separators=(",", ":"), sort_keys=True).encode()
    ok = verify_signature(payload, signature, worker_public_key_file)
    if not ok:
        print(f"verify_handshake_response: signature verification FAILED for worker key {worker_public_key_file}", flush=True)
    else:
        print(f"verify_handshake_response: signature verified for session {session_id}", flush=True)
    return ok

def sign_envelope(payload: dict, session_nonce: str, private_key_path: str) -> dict:
    """Attach session nonce and RSA signature to a message payload."""
    # Include nonce in the signed structure
    payload_with_nonce = dict(payload)
    payload_with_nonce["nonce"] = session_nonce

    msg_bytes = json.dumps(payload_with_nonce, separators=(",", ":"), sort_keys=True).encode()
    sig = sign_message(msg_bytes, private_key_path)

    return {"message": payload_with_nonce, "signature": sig.hex()}

def verify_envelope(envelope: dict, expected_nonce: str, public_key_path: str) -> bool:
    """Verify the message signature and nonce freshness."""
    try:
        msg = envelope["message"]
        sig = bytes.fromhex(envelope["signature"])
    except Exception as e:
        print(f"verify_envelope: malformed envelope: {e}", flush=True)
        return False

    # Nonce check (freshness)
    found_nonce = msg.get("nonce")
    if found_nonce != expected_nonce:
        print(f"verify_envelope: nonce mismatch (expected={expected_nonce} got={found_nonce})", flush=True)
        return False

    msg_bytes = json.dumps(msg, separators=(",", ":"), sort_keys=True).encode()
    ok = verify_signature(msg_bytes, sig, public_key_path)
    if not ok:
        print(f"verify_envelope: signature verification FAILED for public key {public_key_path}", flush=True)
    else:
        print(f"verify_envelope: envelope signature verified (nonce={expected_nonce})", flush=True)
    return ok
