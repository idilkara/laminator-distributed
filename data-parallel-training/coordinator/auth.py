import os
import json
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from binascii import unhexlify


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

