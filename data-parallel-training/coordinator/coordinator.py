# coordinator.py
import argparse
import time
from dataclasses import dataclass
import os, random
import json
import numpy as np
import torch
import torch.nn as nn  # noqa: F401 (kept if your models import needs it)
import zmq
import hashlib

from models import ModelHandler
from data_preprocess import process_census
from training import batches_for_workers, _avg_grads_and_loss

from auth import sign_message, verify_signature, create_handshake_offer, verify_handshake_response, sign_envelope, verify_envelope

# Load CoordinatorConfig to read default endpoints and key paths
from config import CoordinatorConfig


#verify if attestation from worker matches expected attestation
def verfiy_attestation(attestation:dict, expected_attestation:dict) -> bool:
    # Implement attestation verification logic here
    # For example, compare expected values with received attestation
    for key, expected_value in expected_attestation.items():
        received_value = attestation.get(key)
        if received_value != expected_value:
            print(f"Attestation mismatch for {key}: expected {expected_value}, got {received_value}", flush=True)
            return False
    print("Attestation verified successfully.", flush=True)
    return True




# ---------- Training Loop ----------
def train(cfg: CoordinatorConfig):
    
    ctx = zmq.Context.instance()

    session_keys = {}  # wid -> bytes
    pending_offers = {}  # wid -> offer dict (session_id, nonce, signature, pubkey?)

    # ---- data sockets ----
    task_out = ctx.socket(zmq.ROUTER)
    task_out.bind(cfg.TASK_ENDPOINT)

    results_in = ctx.socket(zmq.PULL)
    results_in.bind(cfg.RESULT_ENDPOINT)

    print("Coordinator started (waiting for worker HELLOs for handshake).", flush=True)

    pending = set(str(i) for i in range(cfg.NUM_WORKERS))
    coord_pub_path = os.path.join(os.path.dirname(cfg.PRIVATE_KEY_PATH), "coordinator_public.pem")

    poller = zmq.Poller()
    poller.register(task_out, zmq.POLLIN)
    poller.register(results_in, zmq.POLLIN)

    timeout_seconds = 15
    start = time.time()
    # record handshake start time (covers HELLO/offer/response phases)
    handshake_start = time.time()

    # ---------- Phase 1: wait for HELLO and send offer ----------
    while pending and (time.time() - start) < timeout_seconds:
        events = dict(poller.poll(1000))
        if task_out in events:
            ident, msg_bytes = task_out.recv_multipart()
            try:
                msg = json.loads(msg_bytes.decode())
            except Exception:
                continue

            wid = str(msg.get("wid"))
            if msg.get("type") == "hello" and wid in pending:
                print(f"Coordinator: got HELLO from worker {wid}", flush=True)
                # create and send handshake offer
                session_id = f"sess-{wid}-{int(time.time())}"
                offer = create_handshake_offer(session_id, cfg.PRIVATE_KEY_PATH,
                                            coordinator_public_key_file=coord_pub_path)
                # remember the offer so we can validate the worker's response
                pending_offers[wid] = offer
                env = {"wid": wid, "handshake": offer}
                task_out.send_multipart([ident,
                    json.dumps(env, separators=(",", ":"), sort_keys=True).encode()])
                print(f"Coordinator: sent handshake offer to worker {wid}", flush=True)

    # ---------- Phase 2: collect responses ----------
    print("Coordinator: waiting for handshake responses...", flush=True)
    start = time.time()
    while pending and (time.time() - start) < timeout_seconds:
        events = dict(poller.poll(1000))
        if results_in in events:
            try:
                resp_env = results_in.recv_json()
            except Exception:
                continue

            wid = str(resp_env.get("wid"))
            hresp = resp_env.get("handshake_response")
            if not hresp or wid not in pending:
                continue

            try:
                worker_pub = cfg.WORKER_KEY_PATHS[int(wid)]
            except Exception:
                print(f"Coordinator: no public key for worker {wid}", flush=True)
                pending.discard(wid)
                continue

            expected = pending_offers.get(wid)
            if expected is None:
                print(f"Coordinator: no matching offer for worker {wid}; ignoring response", flush=True)
                pending.discard(wid)
                continue

            ok = verify_handshake_response(
                hresp,
                worker_pub,
                expected_session_id=expected.get("session_id"),
                expected_nonce=expected.get("nonce"),
            )
            if ok:
                # store the expected nonce (from our offer) as the session key
                session_keys[wid] = expected.get("nonce")
                # cleanup
                pending_offers.pop(wid, None)
                pending.discard(wid)
                print(f"Coordinator: handshake completed with worker {wid}", flush=True)
            else:
                print(f"Coordinator: handshake FAILED for worker {wid}", flush=True)

    if pending:
        print(f"Coordinator: handshake timed out for workers {sorted(list(pending))}", flush=True)

    # handshake end/time
    handshake_end = time.time()
    handshake_time = handshake_end - handshake_start
    print(f"Coordinator: handshake phase completed in {handshake_time:.3f}s", flush=True)



    # Data preparation (measure time)
    preprocess_start = time.time()
    X_train, y_train, X_test, y_test = process_census()
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start
    print(f"Coordinator: data preprocessing completed in {preprocess_time:.3f}s", flush=True)
    rng = np.random.default_rng(cfg.SEED)

    # Initialize model (JSON)
    model_string = '''
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
    model = ModelHandler.parse_model_string(model_string)  # ensure same as workers
    assert model != -1

    print(
        f"Coordinator started with {cfg.NUM_WORKERS} workers. "
        f"Data: X={X_train.shape}, y={y_train.shape}",
        flush=True
    )

    # Report structure: report[epoch][wid] = {"status": <str>, "notes": [<str>, ...]}
    report = {}

    
    # ---------- Training epochs ----------
    training_start = time.time()
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        # prepare a per-epoch report mapping
        report.setdefault(epoch, {})

        # expected_hashes keeps the hashes we computed when sending each
        # task so we can verify worker replies later in this epoch.
        expected_hashes = {}

        def _stable_json_hash(obj):
            try:
                j = json.dumps(obj, sort_keys=True, separators=(",",":"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            except Exception:
                j = json.dumps(str(obj), sort_keys=True, separators=(",",":"))
            return hashlib.sha256(j.encode()).hexdigest()

        # Send tasks securely
        for wid, Xb, yb in batches_for_workers(X_train, y_train, cfg.NUM_WORKERS, rng):

            wid_str = str(wid)
            payload = {
                "worker_id": wid_str,
                "X": Xb.tolist(),
                "y": yb.tolist(),
                # Workers expect 'architecture' and 'weights'
                "architecture": model_string,
                "weights": {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
                "lr": float(cfg.LR),
                "epoch": int(epoch)
            }

            # Compute and store stable hashes for this task so we can verify
            # worker responses later.
            H_DTr = _stable_json_hash({"X": payload["X"], "y": payload["y"]})
            H_MAr = _stable_json_hash(payload["architecture"])
            H_Me_init = _stable_json_hash(payload["weights"])
            H_T = _stable_json_hash({"lr": payload["lr"], "epoch": payload["epoch"]})
            expected_hashes[wid_str] = {
                "H_DTr": H_DTr,
                "H_MAr": H_MAr,
                "H_Me_init": H_Me_init,
                "H_T": H_T,
            }

            # initialize per-worker report entry for this epoch
            report[epoch].setdefault(wid_str, {"status": "pending", "notes": []})

            # Serialize and sign the payload, include the session nonce in the signed structure
            session_nonce = session_keys.get(wid_str)
            if not session_nonce:
                print(f"Coordinator: no session nonce for worker {wid_str}; skipping task", flush=True)
                continue
            signed_envelope = sign_envelope(payload, session_nonce, cfg.PRIVATE_KEY_PATH)
            env = {"wid": wid, "payload": signed_envelope}

            # Router expects [identity, payload]; send the JSON payload as a
            # single frame. Worker DEALER socket (with identity set) will
            # receive only the payload frame.
            task_out.send_multipart([wid_str.encode(), json.dumps(env, separators=(",",":"), sort_keys=True).encode()])

    
        # Receive results securely
        grads_accum = []
        losses = []
        counts = []
        for _ in range(cfg.NUM_WORKERS):
            env = results_in.recv_json()
            wid = env["wid"]
            payload_env = env.get("payload")
            if not isinstance(payload_env, dict) or "message" not in payload_env or "signature" not in payload_env:
                print(f"Coordinator: malformed result envelope from worker {wid}", flush=True)
                report.setdefault(epoch, {}).setdefault(str(wid), {"status": "malformed", "notes": []})
                report[epoch][str(wid)]["status"] = "malformed_envelope"
                report[epoch][str(wid)]["notes"].append("missing message or signature in payload")
                continue

            try:
                worker_pub = cfg.WORKER_KEY_PATHS[int(wid)]
            except Exception:
                print(f"Coordinator: no public key for worker {wid}", flush=True)
                report.setdefault(epoch, {}).setdefault(str(wid), {"status": "no_pubkey", "notes": []})
                report[epoch][str(wid)]["status"] = "no_pubkey"
                report[epoch][str(wid)]["notes"].append("no public key configured for worker")
                continue

            # Verify envelope signature and nonce freshness using the established session nonce
            session_nonce = session_keys.get(str(wid))
            if not session_nonce:
                print(f"Coordinator: no session for worker {wid}; ignoring result", flush=True)
                report.setdefault(epoch, {}).setdefault(str(wid), {"status": "no_session", "notes": []})
                report[epoch][str(wid)]["status"] = "no_session"
                report[epoch][str(wid)]["notes"].append("no session nonce for worker")
                continue

            if not verify_envelope(payload_env, session_nonce, worker_pub):
                print(f"Coordinator: envelope verification failed for worker {wid}; ignoring", flush=True)
                report.setdefault(epoch, {}).setdefault(str(wid), {"status": "invalid_signature", "notes": []})
                report[epoch][str(wid)]["status"] = "invalid_signature"
                report[epoch][str(wid)]["notes"].append("envelope signature verification failed or signed by wrong key")
                continue

            # Extract worker payload (remove nonce)
            worker_payload = dict(payload_env["message"])
            worker_payload.pop("nonce", None)

            # Verify that the worker-provided hashes match what we sent.
            expected = expected_hashes.get(str(wid))
            received_hashes = worker_payload.get("hashes") or {}
            if expected is None:
                print(f"Coordinator: no expected hashes recorded for worker {wid}; ignoring result", flush=True)
                report.setdefault(epoch, {}).setdefault(str(wid), {"status": "no_expected_hashes", "notes": []})
                report[epoch][str(wid)]["status"] = "no_expected_hashes"
                report[epoch][str(wid)]["notes"].append("no expected hashes stored for this worker/task")
                continue

            mismatch = False
            for k, v in expected.items():
                if received_hashes.get(k) != v:
                    print(f"Coordinator: hash mismatch from worker {wid} for {k}: expected {v}, got {received_hashes.get(k)}", flush=True)
                    mismatch = True
            if mismatch:
                print(f"Coordinator: ignoring result from worker {wid} due to hash mismatch", flush=True)
                # record mismatch details
                report.setdefault(epoch, {}).setdefault(str(wid), {"status": "hash_mismatch", "notes": []})
                report[epoch][str(wid)]["status"] = "hash_mismatch"
                # record which keys mismatched and values
                for k, v in expected.items():
                    if received_hashes.get(k) != v:
                        report[epoch][str(wid)]["notes"].append(f"{k}: expected {v}, got {received_hashes.get(k)}")
                continue

            # Hashes verified OK — log a concise confirmation and accept result
            print(
                f"Coordinator: hash verification PASSED for worker {wid} | "
                f"H_DTr={expected['H_DTr'][:12]}..., H_MAr={expected['H_MAr'][:12]}..., H_Me_init={expected['H_Me_init'][:12]}..., H_T={expected['H_T'][:12]}...",
                flush=True,
            )

            # record success and accept result
            report.setdefault(epoch, {}).setdefault(str(wid), {"status": "ok", "notes": []})
            report[epoch][str(wid)]["status"] = "ok"
            grads_accum.append(worker_payload["grads"])  # dict[name -> list]
            losses.append(worker_payload["loss"])
            counts.append(worker_payload["n"])

        # Average and apply gradients
        avg_grads, avg_loss = _avg_grads_and_loss(grads_accum, counts, losses)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in avg_grads:
                    continue
                g = avg_grads[name].to(param.device, dtype=param.dtype)
                param -= cfg.LR * g

        dt = time.time() - t0

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_test_t)                     # shape: [N, 2]
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        acc = float(np.mean(y_pred == y_test))

        print(
            f"[Epoch {epoch+1:02d}/{cfg.EPOCHS}] loss={avg_loss:.4f} | time={dt:.2f}s | Acc={acc:.4f}",
            flush=True
        )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_t)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    acc = float(np.mean(y_pred == y_test))
    print(f"Final test Accuracy: {acc:.4f}", flush=True)

    training_end = time.time()
    total_training_time = training_end - training_start
    avg_epoch = total_training_time / max(1, cfg.EPOCHS)
    print(
        f"\nTiming summary: handshake={handshake_time:.3f}s | preprocess={preprocess_time:.3f}s | total_training={total_training_time:.3f}s | avg_epoch={avg_epoch:.3f}s",
        flush=True,
    )

    # Write out a human-readable report of per-epoch per-worker verification results
    try:
        # Prefer writing to the mounted ./data directory so the host can
        # inspect the report when running in Docker. Create the dir if
        # it doesn't exist.
        report_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "hash_report.txt")
        with open(report_path, "w") as rf:
            rf.write(f"Coordinator verification report\n")
            rf.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for e in sorted(report.keys()):
                rf.write(f"Epoch {e}:\n")
                for wid in sorted(report[e].keys(), key=lambda x: int(x)):
                    entry = report[e][wid]
                    status = entry.get("status", "unknown")
                    notes = entry.get("notes", [])
                    rf.write(f"  Worker {wid}: {status}")
                    if notes:
                        rf.write(" -- ")
                        rf.write("; ".join(notes))
                    rf.write("\n")
                rf.write("\n")
        print(f"Coordinator: wrote verification report to {report_path}", flush=True)

        # Also print the report contents to stdout so it's visible in logs
        try:
            with open(report_path, "r") as rf2:
                print("\n----- Verification report -----", flush=True)
                print(rf2.read(), flush=True)
                print("----- End of report -----\n", flush=True)
        except Exception as e:
            print(f"Coordinator: failed to print verification report: {e}", flush=True)
    except Exception as e:
        print(f"Coordinator: failed to write verification report: {e}", flush=True)


    # Send SHUTDOWN control message to all workers so they can exit cleanly.
    try:
        shutdown_payload = {"control": "SHUTDOWN"}
        for wid in range(cfg.NUM_WORKERS):
            wid_str = str(wid)
            session_nonce = session_keys.get(wid_str)
            if not session_nonce:
                print(f"Coordinator: no session for worker {wid_str}; skipping SHUTDOWN", flush=True)
                continue
            signed_envelope = sign_envelope(shutdown_payload, session_nonce, cfg.PRIVATE_KEY_PATH)
            env = {"wid": wid_str, "payload": signed_envelope}
            # Send as multipart [identity, payload] so worker DEALER receives payload frame
            task_out.send_multipart([wid_str.encode(), json.dumps(env, separators=(",",":"), sort_keys=True).encode()])
        print("Coordinator: sent SHUTDOWN to all workers", flush=True)
    except Exception:
        print("Coordinator: failed to send SHUTDOWN to workers", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism (can slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Build a CoordinatorConfig instance and copy runtime parameters onto it.
    cfg = CoordinatorConfig()
    # CoordinatorConfig already has .private_key_path and .worker_key_paths set.
    cfg.NUM_WORKERS = args.num_workers
    cfg.EPOCHS = args.epochs
    cfg.LR = args.lr
    cfg.SEED = seed
    train(cfg)


if __name__ == "__main__":
    main()
