# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def generate_quote(user_data) -> bytes:
    with open('/dev/attestation/user_report_data', 'wb') as f:
        f.write(user_data)
    with open('/dev/attestation/quote', 'rb') as f:
        quote = f.read()
    return quote

def generate_fake_quote(user_data: bytes) -> dict:
    return {
        "quote": "FAKE_QUOTE_BASE64==",
        "report_data": user_data.hex(),
        "mrenclave": "FAKE_MRENCLAVE",
        "mrsigner": "FAKE_MRSIGNER",
        "is_debug": True,
    }


# import struct
# import hashlib
# import os

# def generate_fake_quote(report_data: bytes) -> bytes:
#     """
#     Generate a fake SGX-like quote as bytes for testing only.

#     Layout (little-endian):
#       0..3    : magic b"SGXQ" (4 bytes)
#       4..5    : version (uint16)            -> 3
#       6..7    : signature_type (uint16)     -> 1 (fake/placeholder)
#       8       : is_debug (uint8)            -> 1 if debug else 0
#       9..11   : reserved / padding (3 bytes)
#      12..43   : mrenclave (32 bytes)       -> sha256(report_data + b'mre')
#      44..75   : mrsigner (32 bytes)        -> sha256(report_data + b'mrs')
#      76..79   : report_data_len (uint32)
#      80..(80+N-1) : report_data (N bytes)
#      .. next  : sig_len (uint16)
#      .. next  : signature (sig_len bytes)  -> placeholder (sha256 * 2)

#     Returns:
#       bytes: the binary fake quote
#     """
#     # Parameters
#     MAGIC = b"SGXQ"
#     VERSION = 3             # mimic quote version 3
#     SIGTYPE = 1             # fake signature type
#     IS_DEBUG = 1            # mark as debug (you can set 0 if you prefer)

#     # compute deterministic mrenclave / mrsigner from report_data so same input => same quote
#     mrenclave = hashlib.sha256(report_data + b"mre").digest()  # 32 bytes
#     mrsigner  = hashlib.sha256(report_data + b"mrs").digest()  # 32 bytes

#     # signature placeholder: 64 bytes derived from sha256(report_data + b"sig") repeated/truncated
#     sig_base = hashlib.sha256(report_data + b"sig").digest()
#     signature = sig_base + hashlib.sha256(sig_base).digest()
#     signature = signature[:64]  # ensure 64 bytes

#     # Build header and fields
#     header = bytearray()
#     header += MAGIC
#     header += struct.pack("<H", VERSION)     # uint16
#     header += struct.pack("<H", SIGTYPE)     # uint16
#     header += struct.pack("<B", IS_DEBUG)    # uint8
#     header += b"\x00\x00\x00"                # padding (3 bytes)

#     # Attach mrenclave and mrsigner
#     header += mrenclave
#     header += mrsigner

#     # Report data length and content
#     report_len = len(report_data)
#     header += struct.pack("<I", report_len)  # uint32
#     payload = report_data

#     # signature length and signature bytes
#     sig_len = len(signature)
#     sig_part = struct.pack("<H", sig_len) + signature  # uint16 + sig bytes

#     return bytes(header) + payload + sig_part


# # Example usage
# if __name__ == "__main__":
#     rd = b"example nonce or report-data"
#     q = generate_fake_quote(rd)
#     print("quote length:", len(q))
#     # save to file
#     with open("fake_quote.bin", "wb") as f:
#         f.write(q)
