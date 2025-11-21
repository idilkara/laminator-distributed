#!/usr/bin/env python3

import json
import hashlib
import random
import os
import time
import re

import numpy as np
import torch

from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

from data_preprocess import process_census
from models import ModelHandler


# ===================== SETTINGS =====================

# Paths (make sure these match your coordinator setup)
COORD_PUBLIC_KEY_PATH = os.path.join(".", "coordinator_public.pem")
REPORT_PATH = os.path.join(".", "hash_report.txt")
SIG_PATH = os.path.join(".", "hash_report.txt.sig")

# If you used MODEL_JSON / INITIAL_WEIGHTS when running coordinator, set these.
# Otherwise leave them as None to use the default model & default init.
MODEL_JSON_PATH = None           # e.g. os.path.join("config", "model.json")
INITIAL_WEIGHTS_PATH = None      # e.g. os.path.join("data", "initial_weights.json")

# Training hyperparameters used by the coordinator for this run
LR = 0.1 
EPOCHS = 10
SEED = 42 #DEFAULT
NUM_WORKERS = 2   # <<< set this to the same --num-workers used with coordinator.py


# ===================== CRYPTO HELPERS =====================

def verify_signature(message: bytes, signature: bytes, public_key_file: str) -> bool:
    """Verify an RSA signature (PKCS#1 v1.5 + SHA256)."""
    key = RSA.import_key(open(public_key_file, "rb").read())
    h = SHA256.new(message)
    try:
        pkcs1_15.new(key).verify(h, signature)
        return True
    except (ValueError, TypeError):
        return False


def extract_signature_hex(sig_path: str) -> str:
    """Read the signature_hex line from hash_report.txt.sig."""
    with open(sig_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("signature_hex:"):
                # line format: "signature_hex: <hex>\n"
                return line.split(":", 1)[1].strip()
    raise ValueError(f"signature_hex line not found in {sig_path}")


# ===================== REPORT & HASHING HELPERS =====================

def parse_coordinator_report(report_text: str) -> dict:
    """
    Parse the coordinator verification report and extract key/value pairs like:

      H_dataset: <hex>
      H_arch: <hex>
      H_weights_init: <hex>
      H_config: <hex>
    """
    result = {}
    for line in report_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        result[key] = val
    return result


def _stable_json_hash(obj) -> str:
    """
    Same helper as in coordinator.py:
    json.dumps with sort_keys + compact separators, then SHA256.
    """
    try:
        j = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
    except Exception:
        j = json.dumps(
            str(obj),
            sort_keys=True,
            separators=(",", ":"),
        )
    return hashlib.sha256(j.encode()).hexdigest()


def recompute_global_hashes() -> dict:
    """
    Recompute the same global hashes that coordinator.py writes into hash_report.txt:
      - H_dataset
      - H_arch
      - H_weights_init
      - H_config
    """

    # Seed everything like coordinator.main()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Data preprocessing – same call as in train()
    X_train, y_train, X_test, y_test = process_census()

    # Default model JSON string must match coordinator.py exactly
    default_model_string = '''
    {
        "model_type": "CustomizableLinearNet",
        "params": {
            "hidden_layer_sizes": [128, 256, 128],
            "input_dim": 93,
            "output_dim": 2,
            "activation": "tanh",
            "flatten": true
        }
    }
    '''

    # Choose model_string the same way coordinator.py does
    if MODEL_JSON_PATH:
        try:
            with open(MODEL_JSON_PATH, "r") as mf:
                model_string = mf.read()
        except Exception as e:
            print(f"Verifier: failed to read model JSON {MODEL_JSON_PATH}: {e}; falling back to default")
            model_string = default_model_string
    else:
        model_string = default_model_string

    # Build model in the same way
    model = ModelHandler.parse_model_string(model_string)
    assert model != -1

    # 1) Dataset hash
    H_dataset = _stable_json_hash({
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
    })

    # 2) Architecture hash
    H_arch = _stable_json_hash(model_string)

    # 3) Initial weights hash
    if INITIAL_WEIGHTS_PATH:
        try:
            with open(INITIAL_WEIGHTS_PATH, "r") as wf:
                initial_weights_payload = json.load(wf)
            # load into model like coordinator (not strictly needed for hash, but keeps behavior identical)
            coerced = {k: torch.as_tensor(v) for k, v in initial_weights_payload.items()}
            model.load_state_dict(coerced)
        except Exception as e:
            print(f"Verifier: failed to load initial weights from {INITIAL_WEIGHTS_PATH}: {e}; using model defaults")
            initial_weights_payload = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
    else:
        initial_weights_payload = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}

    H_weights_init = _stable_json_hash(initial_weights_payload)

    # 4) Training configuration hash
    H_config = _stable_json_hash({
        "lr": float(LR),
        "epochs": int(EPOCHS),
        "seed": int(SEED),
        "num_workers": int(NUM_WORKERS),
    })

    return {
        "H_dataset": H_dataset,
        "H_arch": H_arch,
        "H_weights_init": H_weights_init,
        "H_config": H_config,
    }


def compute_final_weights_hash_from_file(path: str) -> str | None:
    """
    Try to load a final_weights JSON file and return its stable json hash.
    Returns None if the file can't be read or parsed.
    """
    if not path:
        return None
    try:
        if not os.path.isabs(path):
            # allow relative paths
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return _stable_json_hash(payload)
    except Exception:
        return None

def parse_epoch_statuses(report_text: str):
    """
    Parse epoch/worker lines of the form:

      Epoch 0:
        Worker 0: ok
        Worker 1: invalid_signature -- ...

    Returns:
      dict[int, list[(worker_id: int, is_ok: bool, raw_status: str)]]
    """
    epoch_results = {}
    current_epoch = None

    for line in report_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match "Epoch N:"
        m_epoch = re.match(r"Epoch\s+(\d+):", line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            epoch_results.setdefault(current_epoch, [])
            continue

        # Match "Worker K: <status ...>" only when we're inside an epoch block
        if current_epoch is not None:
            m_worker = re.match(r"Worker\s+(\d+):\s*(.*)", line)
            if m_worker:
                worker_id = int(m_worker.group(1))
                rest = m_worker.group(2).strip()
                # Take the first token as the status keyword (e.g., "ok", "invalid_signature")
                status_token = rest.split(None, 1)[0].lower() if rest else ""
                is_ok = (status_token == "ok")
                epoch_results[current_epoch].append((worker_id, is_ok, rest))

    return epoch_results

# ===================== MAIN =====================
def main():
    timings = {}
    start_time = time.time()

    # ---- 1) Read report and signature + verify signature ----
    step_start = time.time()
    with open(REPORT_PATH, "r", encoding="utf-8") as rf:
        report_text = rf.read()
    report_bytes = report_text.encode("utf-8")

    sig_hex = extract_signature_hex(SIG_PATH)
    sig_bytes = bytes.fromhex(sig_hex)

    sig_ok = verify_signature(report_bytes, sig_bytes, COORD_PUBLIC_KEY_PATH)
    if sig_ok:
        print("✔ Signature is VALID: hash_report.txt was signed by the coordinator key.")
    else:
        print("✖ Signature is INVALID: hash_report.txt does NOT match the signature/public key.")

    timings["read_and_verify"] = time.time() - step_start
    print(f"[timing] read report and verify signature: {timings['read_and_verify']:.3f}s")

    # ---- 2) Parse hashes from report ----
    step_start = time.time()
    report_hashes = parse_coordinator_report(report_text)
    timings["parse_report"] = time.time() - step_start
    print(f"[timing] parse coordinator report: {timings['parse_report']:.3f}s")

    # ---- 3) Recompute global hashes with same logic & hardcoded params ----
    step_start = time.time()
    recomputed = recompute_global_hashes()
    timings["recompute_hashes"] = time.time() - step_start
    print(f"[timing] recompute global hashes: {timings['recompute_hashes']:.3f}s")

    # ---- 4) Attempt to locate and hash final_weights.json ----
    step_start = time.time()

    final_hash = None
    reported_final_path = report_hashes.get("final_weights_path")

    # Try the exact path the coordinator reported (if any)
    if reported_final_path and reported_final_path != "<unavailable>":
        final_hash = compute_final_weights_hash_from_file(reported_final_path)

    # If that failed, try likely local locations
    if final_hash is None:
        candidates = [
            os.path.join(os.getcwd(), "final_weights.json"),
            os.path.join(os.getcwd(), "data", "final_weights.json"),
            os.path.join(os.path.dirname(REPORT_PATH) or ".", "final_weights.json"),
            "final_weights.json",
        ]
        for c in candidates:
            final_hash = compute_final_weights_hash_from_file(c)
            if final_hash is not None:
                break

    timings["final_weights_hash"] = time.time() - step_start
    print(f"[timing] compute final_weights hash (file lookup): {timings['final_weights_hash']:.3f}s")

    # Attach final weights hash to recomputed dict so we can compare uniformly
    recomputed["H_weights_final"] = final_hash

    # ---- 5) Compare global hashes ----
    print("\nGlobal hash comparison (report vs recomputed):")
    all_hashes_ok = True
    for key in ["H_dataset", "H_arch", "H_weights_init", "H_config", "H_weights_final"]:
        rep_val = report_hashes.get(key)
        rec_val = recomputed.get(key)
        if rep_val == rec_val and rep_val is not None:
            print(f"  ✔ {key} MATCHES")
        else:
            print(f"  ✖ {key} MISMATCH")
            print(f"     report:     {rep_val}")
            print(f"     recomputed: {rec_val}")
            all_hashes_ok = False

    if all_hashes_ok:
        print("\n✔ All coordinator global hashes match (H_dataset, H_arch, H_weights_init, H_config, H_weights_final).")
    else:
        print("\n✖ One or more global hashes do not match. Check hardcoded params and inputs.")

    # ---- 6) Parse epoch / worker statuses ----
    step_start = time.time()
    epoch_statuses = parse_epoch_statuses(report_text)
    timings["parse_epochs"] = time.time() - step_start
    print(f"[timing] parse epoch/worker statuses: {timings['parse_epochs']:.3f}s")

    epochs_ok = True
    failing_epochs = set()

    for epoch, entries in epoch_statuses.items():
        for worker_id, is_ok, raw_status in entries:
            if not is_ok:
                epochs_ok = False
                failing_epochs.add(epoch)

    if epochs_ok:
        print("\n✔ All epoch/worker statuses are ok.")
    else:
        failing_epochs_list = sorted(failing_epochs)
        print("\n✖ One or more epochs contain non-ok worker statuses.")
        print(f"   failing_epochs: {', '.join(str(e) for e in failing_epochs_list)}")

    # ---- 7) Overall timing and result summary ----
    total = time.time() - start_time
    timings["total"] = total
    print(f"[timing] total verifier runtime: {total:.3f}s")

    # Timing summary line with all timings
    print(
        "timing summary: "
        f"total={timings['total']:.3f}s, "
        f"read_and_verify={timings['read_and_verify']:.3f}s, "
        f"parse_report={timings['parse_report']:.3f}s, "
        f"recompute_hashes={timings['recompute_hashes']:.3f}s, "
        f"final_weights={timings['final_weights_hash']:.3f}s, "
        f"parse_epochs={timings['parse_epochs']:.3f}s"
    )

    overall_ok = sig_ok and all_hashes_ok and epochs_ok
    print(
        f"summary of timing and results: "
        f"signature_ok={sig_ok} "
        f"hashes_ok={all_hashes_ok} "
        f"epochs_ok={epochs_ok} "
        f"overall_ok={overall_ok}"
    )

    # Exit code policy: ANY failure (signature, hashes, or epoch statuses) -> non-zero exit
    if not overall_ok:
        exit(1)


if __name__ == "__main__":
    main()
