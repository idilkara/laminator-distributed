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
    key = RSA.import_key(open(private_key_file, "rb").read())
    h = SHA256.new(message)
    signature = pkcs1_15.new(key).sign(h)
    return signature


def verify_signature(message: bytes, signature: bytes, public_key_file: str) -> bool:
    key = RSA.import_key(open(public_key_file, "rb").read())
    h = SHA256.new(message)
    try:
        pkcs1_15.new(key).verify(h, signature)
        return True
    except (ValueError, TypeError):
        return False


def verify_handshake_offer(offer: dict, coordinator_public_key_file: str) -> bool:
    """Verify a coordinator handshake offer signature and basic structure.
    """
    try:
        session_id = offer["session_id"]
        nonce = offer["nonce"]
        signature = bytes.fromhex(offer["signature"])
    except Exception as e:
        print(f"worker.verify_handshake_offer: malformed offer: {e}", flush=True)
        return False
    payload = json.dumps({"session_id": session_id, "nonce": nonce}, separators=(",", ":"), sort_keys=True).encode()
    # If coordinator included its pubkey in the offer, prefer that (avoids stale local copies)
    pubkey_pem = offer.get("pubkey")
    if pubkey_pem:
        # verify using the provided public key material
        try:
            key = RSA.import_key(pubkey_pem.encode())
            h = SHA256.new(payload)
            pkcs1_15.new(key).verify(h, signature)
            print(f"worker.verify_handshake_offer: signature verified using embedded pubkey for session {session_id}", flush=True)
            return True
        except Exception as e:
            print(f"worker.verify_handshake_offer: signature verification FAILED with embedded pubkey: {e}", flush=True)
            return False
    # Fallback to local file path
    ok = verify_signature(payload, signature, coordinator_public_key_file)
    if ok:
        print(f"worker.verify_handshake_offer: signature verified using local pubkey for session {session_id}", flush=True)
    else:
        print(f"worker.verify_handshake_offer: signature verification FAILED using local pubkey for session {session_id}", flush=True)
    return ok


def create_handshake_response(offer: dict, worker_id: str, worker_private_key_file: str) -> dict:
    """Create a worker-side handshake response that proves possession of the worker key.

    Mirrors the coordinator-side verifier.
    """
    # Use canonical JSON so coordinator can reproduce the exact bytes to verify
    ts_val = int(time.time())
    payload = json.dumps({
        "session_id": offer["session_id"],
        "nonce": offer["nonce"],
        "worker_id": worker_id,
        "ts": ts_val,
    }, separators=(",", ":"), sort_keys=True).encode()
    signature = sign_message(payload, worker_private_key_file)
    return {
        "session_id": offer["session_id"],
        "worker_id": worker_id,
        "nonce": offer["nonce"],
        "signature": signature.hex(),
        "ts": ts_val,
    }


def sign_envelope(payload: dict, session_nonce: str, private_key_path: str) -> dict:
    """Attach session nonce and RSA signature to a message payload."""
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
        print(f"worker.verify_envelope: malformed envelope: {e}", flush=True)
        return False
    # Nonce check
    found = msg.get("nonce")
    if found != expected_nonce:
        print(f"worker.verify_envelope: nonce mismatch (expected={expected_nonce} got={found})", flush=True)
        return False
    msg_bytes = json.dumps(msg, separators=(",", ":"), sort_keys=True).encode()
    ok = verify_signature(msg_bytes, sig, public_key_path)
    if ok:
        print(f"worker.verify_envelope: envelope signature verified (nonce={expected_nonce})", flush=True)
    else:
        print(f"worker.verify_envelope: signature verification FAILED for public key {public_key_path}", flush=True)
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
    msg = envelope["message"]
    sig = bytes.fromhex(envelope["signature"])

    # Nonce check (freshness)
    if msg.get("nonce") != expected_nonce:
        print("Nonce mismatch: possible replay or wrong session")
        return False

    msg_bytes = json.dumps(msg, separators=(",", ":"), sort_keys=True).encode()
    return verify_signature(msg_bytes, sig, public_key_path)
