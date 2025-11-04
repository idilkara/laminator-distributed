# auth.py
import base64, hashlib, hmac, json, os, time
from typing import Dict, Tuple, Optional

# --------- utils ---------
def b64e(b: bytes) -> str: return base64.b64encode(b).decode("ascii")
def b64d(s: str) -> bytes: return base64.b64decode(s.encode("ascii"))

def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """Simple HKDF for key derivation (RFC 5869)."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

# --------- handshake (REQ/REP) ---------
# Use one shared "provisioning" secret to *derive* per-session keys.
# Provisioning secret comes from env: HMAC_PROVISIONING_SECRET (32+ bytes recommended).

def derive_session_key(prov_secret: bytes, nonce_w: bytes, nonce_c: bytes, worker_id: str) -> bytes:
    ikm = nonce_w + nonce_c
    info = b"session-hmac:" + worker_id.encode()
    # salt is provisioning secret; hkdf gives forward-secrecy per-session nonces
    return hkdf_sha256(ikm=ikm, salt=prov_secret, info=info, length=32)

def coordinator_handle_handshake(msg: dict, prov_secret: bytes) -> Tuple[dict, Optional[Tuple[str, bytes]]]:
    """
    Input msg: {"type":"hello","worker_id":..., "nonce_w": b64}
    Returns (reply_dict, (worker_id, session_key)) when ready; otherwise (reply_dict, None)
    """
    t = msg.get("type")
    if t == "hello":
        worker_id = str(msg["worker_id"])
        nonce_w = b64d(msg["nonce_w"])
        nonce_c = os.urandom(32)
        # Return challenge with coordinator nonce
        reply = {"type": "challenge", "nonce_c": b64e(nonce_c), "ts": int(time.time())}
        # We’ll verify in the next step; stash data in the reply (stateless approach: echo back a token)
        token = b64e(hmac.new(prov_secret, nonce_w + nonce_c + worker_id.encode(), hashlib.sha256).digest())
        reply["token"] = token
        # No session key yet
        return reply, None

    elif t == "ack":
        # Validate ack and finalize session
        worker_id = str(msg["worker_id"])
        nonce_w = b64d(msg["nonce_w"])
        nonce_c = b64d(msg["nonce_c"])
        token = msg["token"]
        expected_token = b64e(hmac.new(prov_secret, nonce_w + nonce_c + worker_id.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(token, expected_token):
            return {"type": "error", "error": "bad token"}, None
        session_key = derive_session_key(prov_secret, nonce_w, nonce_c, worker_id)
        # Verify ACK MAC to prove the worker derived the same key
        mac = b64d(msg["mac"])
        expected_mac = hmac.new(session_key, b"ACK" + nonce_w + nonce_c, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected_mac):
            return {"type": "error", "error": "bad mac"}, None
        return {"type": "ok"}, (worker_id, session_key)

    else:
        return {"type": "error", "error": "bad type"}, None


def worker_start_handshake(send_req, recv_rep, worker_id: str, prov_secret_present: bool = True) -> bytes:
    """
    Worker side: call send_req(dict) to send to coordinator's REP; recv_rep() to receive replies.
    Returns session_key (bytes).
    """
    # Step 1: hello
    nonce_w = os.urandom(32)
    send_req({"type": "hello", "worker_id": worker_id, "nonce_w": b64e(nonce_w)})
    rep = recv_rep()
    if rep.get("type") != "challenge":
        raise RuntimeError(f"Handshake failed: {rep}")
    nonce_c = b64d(rep["nonce_c"])
    token = rep["token"]
    # Derive session key locally
    # (We don't need provisioning secret locally if we trust the coordinator to issue token,
    # but using the same hkdf derivation gives both sides the same key.)
    # Here we DON’T use the token for derivation; it’s only for coordinator statelessness.
    # Both sides derive with HKDF(nonce_w || nonce_c, prov_secret).
    if not prov_secret_present:
        # If you *really* don't want provisioning secret on the worker, you could instead
        # derive via token + nonces. For simplicity, expect the secret on both ends.
        raise RuntimeError("Worker missing provisioning secret; set HMAC_PROVISIONING_SECRET")
    prov_secret = os.environ.get("HMAC_PROVISIONING_SECRET", "").encode()
    if len(prov_secret) < 16:
        raise RuntimeError("HMAC_PROVISIONING_SECRET too short")
    session_key = derive_session_key(prov_secret, nonce_w, nonce_c, worker_id)
    # Step 2: ack
    mac = hmac.new(session_key, b"ACK" + nonce_w + nonce_c, hashlib.sha256).digest()
    send_req({
        "type": "ack",
        "worker_id": worker_id,
        "nonce_w": b64e(nonce_w),
        "nonce_c": b64e(nonce_c),
        "token": token,
        "mac": b64e(mac),
    })
    rep2 = recv_rep()
    if rep2.get("type") != "ok":
        raise RuntimeError(f"Handshake failed: {rep2}")
    return session_key

# --------- signed message envelopes ---------
def sign_envelope(session_key: bytes, worker_id: str, payload: dict) -> dict:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ts = int(time.time())
    nonce = os.urandom(16)
    to_mac = b"|".join([
        worker_id.encode(),
        str(ts).encode(),
        nonce,
        hashlib.sha256(body).digest(),
    ])
    mac = hmac.new(session_key, to_mac, hashlib.sha256).digest()
    return {
        "wid": worker_id,
        "ts": ts,
        "nonce": b64e(nonce),
        "body": b64e(body),
        "mac": b64e(mac),
        "alg": "HMAC-SHA256",
    }

def verify_envelope(envelope: dict, session_key: bytes, max_skew: int = 120, seen_nonces: Optional[set] = None) -> dict:
    wid = envelope["wid"]
    ts = int(envelope["ts"])
    nonce = b64d(envelope["nonce"])
    body = b64d(envelope["body"])
    mac = b64d(envelope["mac"])

    # Time window check (anti-replay)
    now = int(time.time())
    if abs(now - ts) > max_skew:
        raise RuntimeError("stale or future-dated message")

    # Nonce replay check
    if seen_nonces is not None:
        key = (wid, nonce)
        if key in seen_nonces:
            raise RuntimeError("replay detected")
        # Keep set from growing unbounded in your app; e.g., purge old entries periodically.
        seen_nonces.add(key)

    to_mac = b"|".join([
        wid.encode(),
        str(ts).encode(),
        nonce,
        hashlib.sha256(body).digest(),
    ])
    exp_mac = hmac.new(session_key, to_mac, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, exp_mac):
        raise RuntimeError("bad mac")

    return json.loads(body.decode())
