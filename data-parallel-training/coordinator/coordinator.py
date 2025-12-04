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

    pending_hellos = set(str(i) for i in range(cfg.NUM_WORKERS))
    awaiting_responses = set()
    coord_pub_path = os.path.join(os.path.dirname(cfg.PRIVATE_KEY_PATH), "coordinator_public.pem")

    poller = zmq.Poller()
    poller.register(results_in, zmq.POLLIN)

    timeout_seconds = 15
    loop_start = time.time()
    # record handshake start time (covers HELLO/offer/response phases)
    handshake_start = loop_start

    print("Coordinator: waiting for handshake responses...", flush=True)
    while (pending_hellos or awaiting_responses) and (time.time() - loop_start) < timeout_seconds:
        # Check for HELLOs from workers (ROUTER socket)
        try:
            ident, msg_bytes = task_out.recv_multipart(zmq.NOBLOCK)
            try:
                msg = json.loads(msg_bytes.decode())
            except Exception:
                msg = None

            if isinstance(msg, dict):
                wid = str(msg.get("wid"))
                if msg.get("type") == "hello" and wid in pending_hellos:
                    print(f"Coordinator: got HELLO from worker {wid}", flush=True)
                    session_id = f"sess-{wid}-{int(time.time())}"
                    offer = create_handshake_offer(
                        session_id,
                        cfg.PRIVATE_KEY_PATH,
                        coordinator_public_key_file=coord_pub_path,
                    )
                    pending_offers[wid] = offer
                    env = {"wid": wid, "handshake": offer}
                    task_out.send_multipart([
                        ident,
                        json.dumps(env, separators=(",", ":"), sort_keys=True).encode(),
                    ])
                    awaiting_responses.add(wid)
                    pending_hellos.discard(wid)
                    print(f"Coordinator: sent handshake offer to worker {wid}", flush=True)
        except zmq.Again:
            pass

        # Check for handshake responses from workers (PULL socket)
        events = dict(poller.poll(100))
        if results_in in events:
            try:
                resp_env = results_in.recv_json()
            except Exception:
                continue

            wid = str(resp_env.get("wid"))
            hresp = resp_env.get("handshake_response")
            if not hresp or wid not in awaiting_responses:
                continue

            try:
                print(cfg.WORKER_KEY_PATHS)
                worker_pub = cfg.WORKER_KEY_PATHS[int(wid)]
            except Exception:
                print(f"Coordinator: no public key for worker {wid}", flush=True)
                awaiting_responses.discard(wid)
                continue

            expected = pending_offers.get(wid)
            if expected is None:
                print(f"Coordinator: no matching offer for worker {wid}; ignoring response", flush=True)
                awaiting_responses.discard(wid)
                continue

            ok = verify_handshake_response(
                hresp,
                worker_pub,
                expected_session_id=expected.get("session_id"),
                expected_nonce=expected.get("nonce"),
            )
            if ok:
                session_keys[wid] = expected.get("nonce")
                pending_offers.pop(wid, None)
                awaiting_responses.discard(wid)
                print(f"Coordinator: handshake completed with worker {wid}", flush=True)
            else:
                print(f"Coordinator: handshake FAILED for worker {wid}", flush=True)

    if pending_hellos or awaiting_responses:
        still_pending = sorted(list(pending_hellos.union(awaiting_responses)), key=int)
        print(f"Coordinator: handshake timed out for workers {still_pending}", flush=True)

    handshake_end = time.time()
    handshake_time = handshake_end - handshake_start
    print(f"Coordinator: handshake phase completed in {handshake_time:.3f}s", flush=True)

########### HANDSHAKE ESTABLISHED — PROCEED TO TRAINING ###########

    # Data preparation
    preprocess_start = time.time()
    X_train, y_train, X_test, y_test = process_census()
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start
    print(f"Coordinator: data preprocessing completed in {preprocess_time:.3f}s", flush=True)
    rng = np.random.default_rng(cfg.SEED)

    # Initialize model (JSON). Allow overriding via cfg.MODEL_JSON file path.
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

    # If a model JSON file path was provided, load it; otherwise use default
    if getattr(cfg, "MODEL_JSON", None):
        try:
            with open(cfg.MODEL_JSON, "r") as mf:
                model_string = mf.read()
        except Exception as e:
            print(f"Coordinator: failed to read model JSON {cfg.MODEL_JSON}: {e}; falling back to default", flush=True)
            model_string = default_model_string
    else:
        model_string = default_model_string

    model = ModelHandler.parse_model_string(model_string)  # ensure same as workers
    assert model != -1

    print(
        f"Coordinator started with {cfg.NUM_WORKERS} workers. "
        f"Data: X={X_train.shape}, y={y_train.shape}",
        flush=True
    )

    # Compute and record global hashes: dataset, architecture, initial weights, training config
    def _stable_json_hash(obj):
        try:
            j = json.dumps(obj, sort_keys=True, separators=(",",":"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except Exception:
            j = json.dumps(str(obj), sort_keys=True, separators=(",",":"))
        return hashlib.sha256(j.encode()).hexdigest()

    # Hash the full training dataset (may be large) AND architecture
    H_dataset = _stable_json_hash({"X_train": X_train.tolist(), "y_train": y_train.tolist()})
    H_arch = _stable_json_hash(model_string)

    # If initial weights file provided, load and use; otherwise use model.state_dict()
    initial_weights_payload = None
    if getattr(cfg, "INITIAL_WEIGHTS", None):
        try:
            with open(cfg.INITIAL_WEIGHTS, "r") as wf:
                initial_weights_payload = json.load(wf)
            # load into model to ensure consistency
            coerced = {k: torch.as_tensor(v) for k, v in initial_weights_payload.items()}
            model.load_state_dict(coerced)
        except Exception as e:
            print(f"Coordinator: failed to load initial weights from {cfg.INITIAL_WEIGHTS}: {e}; using model defaults", flush=True)
            initial_weights_payload = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
    else:
        initial_weights_payload = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}

    H_weights_init = _stable_json_hash(initial_weights_payload)

    # Hash training configuration (lr, epochs, seed, num_workers)
    H_config = _stable_json_hash({"lr": float(cfg.LR), "epochs": int(cfg.EPOCHS), "seed": int(cfg.SEED), "num_workers": int(cfg.NUM_WORKERS)})

    # Report structure: report[epoch][wid] = {"status": <str>, "notes": [<str>, ...]}
    report = {}

    ########### TRAINING EPOCHS ###########
    training_start = time.time()

    total_hashing_time_per_task = 0.0
    total_verify_env_time_per_task = 0.0
    total_verify_hash_time_per_task = 0.0
    total_signing_time_per_task = 0.0

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


            hashing_time_for_task_start = time.time()
            # Compute and store stable hashes for this task so we can verify
            # worker responses later.
            H_DTr = _stable_json_hash({"X": payload["X"], "y": payload["y"]})
            H_MAr = H_arch  # architecture hash is global
            H_Me_init = _stable_json_hash(payload["weights"])
            H_T = _stable_json_hash({"lr": payload["lr"], "epoch": payload["epoch"]})
            expected_hashes[wid_str] = {
                "H_DTr": H_DTr,
                "H_MAr": H_MAr,
                "H_Me_init": H_Me_init, # weigths sent in task 
                "H_T": H_T,
            }

            hashing_time_for_task_end = time.time()
            hashing_time_for_task = hashing_time_for_task_end - hashing_time_for_task_start
            total_hashing_time_per_task += hashing_time_for_task


            # initialize per-worker report entry for this epoch
            report[epoch].setdefault(wid_str, {"status": "pending", "notes": []})

            signing_time_for_task_start = time.time()
            # Serialize and sign the payload, include the session nonce in the signed structure
            session_nonce = session_keys.get(wid_str)
            if not session_nonce:
                print(f"Coordinator: no session nonce for worker {wid_str}; skipping task", flush=True)
                continue
            signed_envelope = sign_envelope(payload, session_nonce, cfg.PRIVATE_KEY_PATH)
            env = {"wid": wid, "payload": signed_envelope}
            signing_time_for_task_end = time.time()
            signing_time_for_task = signing_time_for_task_end - signing_time_for_task_start
            total_signing_time_per_task += signing_time_for_task


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

            verify_env_time_per_task_start = time.time()

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
            


            verify_env_time_per_task_end = time.time()
            verify_env_time_per_task = verify_env_time_per_task_end - verify_env_time_per_task_start
            total_verify_env_time_per_task += verify_env_time_per_task

            # Extract worker payload (remove nonce)
            worker_payload = dict(payload_env["message"])
            worker_payload.pop("nonce", None)

            verify_hash_time_for_task_start = time.time()

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

            
            # verify that worker epoch matches the current epoch

            worker_epoch = worker_payload.get("epoch")
            if worker_epoch != epoch:
                print(f"Coordinator: epoch mismatch from worker {wid}: expected {epoch}, got {worker_epoch}", flush=True)
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


            verify_hash_time_for_task_end = time.time()
            verify_hash_time_for_task = verify_hash_time_for_task_end - verify_hash_time_for_task_start
            total_verify_hash_time_per_task += verify_hash_time_for_task
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

    # Prepare final weights payload and hash so we can include it in the report
    try:
        final_weights_payload = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
        H_weights_final = _stable_json_hash(final_weights_payload)
    except Exception:
        final_weights_payload = None
        H_weights_final = None

    # Write out a human-readable report of per-epoch per-worker verification results
    # Measure report generation time (includes writing the report file, final_weights.json and signing)
    report_gen_start = time.time()
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
            rf.write("Global hashes:\n")
            rf.write(f"  H_dataset: {H_dataset}\n")
            rf.write(f"  H_arch: {H_arch}\n")
            rf.write(f"  H_weights_init: {H_weights_init}\n")
            # include final weights hash/path if available
            if H_weights_final is not None:
                rf.write(f"  H_weights_final: {H_weights_final}\n")
                rf.write(f"  weights_final: {final_weights_payload}\n")
            else:
                rf.write(f"  H_weights_final: <unavailable>\n")
            # write final accuracy and final loss (if available)
            try:
                rf.write(f"  final_accuracy: {acc:.6f}\n")
            except Exception:
                rf.write(f"  final_accuracy: <unavailable>\n")
            try:
                rf.write(f"  final_loss: {avg_loss:.6f}\n")
            except Exception:
                rf.write(f"  final_loss: <unavailable>\n")
            # write path for final weights file
            final_weights_path = os.path.join(report_dir, "final_weights.json")
            if final_weights_payload is not None:
                rf.write(f"  final_weights_path: {final_weights_path}\n\n")
            else:
                rf.write(f"  final_weights_path: <unavailable>\n\n")
            rf.write(f"  H_config: {H_config}\n\n")
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
                contents = rf2.read()
                # print(contents, flush=True)
                # Also print concise global hashes line for quick visibility
                try:
                    print(
                        f"Global hashes summary: H_dataset={H_dataset}, H_arch={H_arch}, H_weights_init={H_weights_init}, H_config={H_config}",
                        flush=True,
                    )
                except Exception:
                    pass
                # print("----- End of report -----\n", flush=True)
        except Exception as e:
            print(f"Coordinator: failed to print verification report: {e}", flush=True)

        # Sign the report contents and write signature to a separate file
        try:
            # Before signing the report, also write the final weights JSON (if available)
            if final_weights_payload is not None:
                try:
                    with open(final_weights_path, "w") as fwf:
                        json.dump(final_weights_payload, fwf, separators=(",", ":"), sort_keys=True)
                    print(f"Coordinator: wrote final weights to {final_weights_path}", flush=True)
                except Exception as e:
                    print(f"Coordinator: failed to write final weights file: {e}", flush=True)

            # Use the coordinator private key to sign the human-readable
            # report. The sign_message() function returns raw signature bytes.
            sig_bytes = sign_message(contents.encode(), cfg.PRIVATE_KEY_PATH)
            sig_hex = sig_bytes.hex()
            sig_path = os.path.join(report_dir, "hash_report.txt.sig")
            with open(sig_path, "w") as sf:
                # Write signature as hex along with generation timestamp for convenience
                sf.write(f"signature_hex: {sig_hex}\n")
                sf.write(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                sf.write(f"signed_by_private_key: {os.path.basename(cfg.PRIVATE_KEY_PATH)}\n")
            print(f"Coordinator: wrote report signature to {sig_path}", flush=True)
            # Report generation finished (includes final weights write + signature)
            report_gen_end = time.time()
            report_gen_time = report_gen_end - report_gen_start

            print(f"\nTiming summary: handshake={handshake_time:.3f}s | preprocess={preprocess_time:.3f}s | total_training={total_training_time:.3f}s | avg_epoch={avg_epoch:.3f}s | report_generation={report_gen_time:.3f}s",flush=True)
            print(f"Coordinator: total and average hashing time per task: {total_hashing_time_per_task:.3f},  {total_hashing_time_per_task/( cfg.NUM_WORKERS * cfg.EPOCHS)  } s", flush=True)
            print(f"Coordinator: total and average envelope verification time per task: {total_verify_env_time_per_task:.3f}, {total_verify_env_time_per_task/( cfg.NUM_WORKERS * cfg.EPOCHS)  } s", flush=True)
            print(f"Coordinator: total and average hash verification time per task: {total_verify_hash_time_per_task:.3f}, {total_verify_hash_time_per_task/( cfg.NUM_WORKERS * cfg.EPOCHS)  } s", flush=True)
            print(f"Coordinator: total and average envelope signing time per task: {total_signing_time_per_task:.3f}, {total_signing_time_per_task/( cfg.NUM_WORKERS * cfg.EPOCHS)  } s", flush=True)

            print(f"Coordinator: report generation time: {report_gen_time:.3f}s", flush=True)
        except Exception as e:
            print(f"Coordinator: failed to sign/write report signature: {e}", flush=True)
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
    p.add_argument("--model-json", type=str, default=None, help="Optional path to a model JSON file describing architecture")
    p.add_argument("--initial-weights", type=str, default=None, help="Optional path to a JSON file with initial weights {param_name: nested lists}")
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
    # attach optional inputs to cfg so train() can use them
    cfg.MODEL_JSON = args.model_json
    cfg.INITIAL_WEIGHTS = args.initial_weights
    train(cfg)


if __name__ == "__main__":
    main()
