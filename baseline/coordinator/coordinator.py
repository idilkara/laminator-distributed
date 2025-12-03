# baseline_coordinator_router.py
import argparse
import time
from dataclasses import dataclass
import os, random
import json
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
import zmq
import secrets

from models import ModelHandler
from data_preprocess import process_census
from training import batches_for_workers, _avg_grads_and_loss
from config import CoordinatorConfig


def train(cfg: CoordinatorConfig):
    ctx = zmq.Context.instance()

    # Per-worker session nonce (filled after handshake)
    session_nonces = {}  # wid_str -> nonce

    # ---- data sockets ----
    task_out = ctx.socket(zmq.ROUTER)
    task_out.bind(cfg.TASK_ENDPOINT)

    results_in = ctx.socket(zmq.PULL)
    results_in.bind(cfg.RESULT_ENDPOINT)

    print("Coordinator started (ROUTER/DEALER baseline, per-worker nonces, no signatures).", flush=True)

    # ---- Handshake phase ----
    pending_hellos = set(str(i) for i in range(cfg.NUM_WORKERS))
    awaiting_responses = set()

    poller = zmq.Poller()
    poller.register(task_out, zmq.POLLIN)
    poller.register(results_in, zmq.POLLIN)

    timeout_seconds = 15
    loop_start = time.time()
    handshake_start = loop_start

    print("Coordinator: waiting for worker HELLOs and handshake responses...", flush=True)
    pending_offers = {}  # wid_str -> offer dict

    while (pending_hellos or awaiting_responses) and (time.time() - loop_start) < timeout_seconds:
        events = dict(poller.poll(1000))

        # ROUTER side: receive HELLO from workers and send offers
        if task_out in events:
            ident, msg_bytes = task_out.recv_multipart()
            try:
                msg = json.loads(msg_bytes.decode())
            except Exception:
                msg = None

            if not isinstance(msg, dict):
                continue

            wid = str(msg.get("wid"))
            if msg.get("type") == "hello" and wid in pending_hellos:
                print(f"Coordinator: got HELLO from worker {wid}", flush=True)
                # Generate per-worker nonce + session_id
                nonce = secrets.token_hex(16)
                session_id = f"sess-{wid}-{int(time.time())}"
                session_nonces[wid] = nonce

                offer = {
                    "session_id": session_id,
                    "nonce": nonce,
                }
                pending_offers[wid] = offer
                env = {"wid": wid, "handshake": offer}
                task_out.send_multipart([
                    ident,
                    json.dumps(env, separators=(",", ":"), sort_keys=True).encode(),
                ])
                awaiting_responses.add(wid)
                pending_hellos.discard(wid)
                print(
                    f"Coordinator: sent handshake offer to worker {wid} "
                    f"(session_id={session_id}, nonce={nonce})",
                    flush=True,
                )

        # PULL side: receive handshake responses
        if results_in in events:
            try:
                resp_env = results_in.recv_json()
            except Exception:
                continue

            wid = str(resp_env.get("wid"))
            hresp = resp_env.get("handshake_response")
            if not hresp or wid not in awaiting_responses:
                continue

            expected = pending_offers.get(wid)
            if expected is None:
                print(f"Coordinator: no matching offer for worker {wid}; ignoring handshake response", flush=True)
                awaiting_responses.discard(wid)
                continue

            # Baseline: just check session_id and nonce match
            if (
                hresp.get("session_id") == expected.get("session_id")
                and hresp.get("nonce") == expected.get("nonce")
                and hresp.get("ok") is True
            ):
                print(
                    f"Coordinator: handshake completed with worker {wid} "
                    f"(session_id={expected['session_id']}, nonce={expected['nonce']})",
                    flush=True,
                )
                awaiting_responses.discard(wid)
                pending_offers.pop(wid, None)
            else:
                print(f"Coordinator: handshake FAILED / mismatched for worker {wid}", flush=True)
                awaiting_responses.discard(wid)
                pending_offers.pop(wid, None)

    if pending_hellos or awaiting_responses:
        still_pending = sorted(list(pending_hellos.union(awaiting_responses)), key=int)
        print(f"Coordinator: handshake timed out for workers {still_pending}", flush=True)

    handshake_end = time.time()
    handshake_time = handshake_end - handshake_start
    print(f"Coordinator: handshake phase completed in {handshake_time:.3f}s", flush=True)

    # ---- Data preparation ----
    preprocess_start = time.time()
    X_train, y_train, X_test, y_test = process_census()
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start
    print(f"Coordinator: data preprocessing completed in {preprocess_time:.3f}s", flush=True)
    rng = np.random.default_rng(cfg.SEED)

    # Initialize model (JSON). Allow overriding via cfg.MODEL_JSON path.
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
    if getattr(cfg, "MODEL_JSON", None):
        try:
            with open(cfg.MODEL_JSON, "r") as mf:
                model_string = mf.read()
        except Exception as e:
            print(f"Coordinator: failed to read model JSON {cfg.MODEL_JSON}: {e}; falling back to default", flush=True)
            model_string = default_model_string
    else:
        model_string = default_model_string

    model = ModelHandler.parse_model_string(model_string)
    assert model != -1

    # Optional: load initial weights (not security-related)
    if getattr(cfg, "INITIAL_WEIGHTS", None):
        try:
            with open(cfg.INITIAL_WEIGHTS, "r") as wf:
                initial_weights_payload = json.load(wf)
            coerced = {k: torch.as_tensor(v) for k, v in initial_weights_payload.items()}
            model.load_state_dict(coerced)
            print(f"Coordinator: loaded initial weights from {cfg.INITIAL_WEIGHTS}", flush=True)
        except Exception as e:
            print(f"Coordinator: failed to load initial weights from {cfg.INITIAL_WEIGHTS}: {e}; using model defaults", flush=True)

    print(
        f"Coordinator started training with {cfg.NUM_WORKERS} workers. "
        f"Data: X={X_train.shape}, y={y_train.shape}",
        flush=True,
    )

    # ---- Training epochs ----
    training_start = time.time()
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        # Send tasks securely-ish: just include per-worker nonce
        num_tasks = 0
        for wid, Xb, yb in batches_for_workers(X_train, y_train, cfg.NUM_WORKERS, rng):
            wid_str = str(wid)
            payload = {
                "worker_id": wid_str,
                "X": Xb.tolist(),
                "y": yb.tolist(),
                "architecture": model_string,
                "weights": {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
                "lr": float(cfg.LR),
                "epoch": int(epoch),
                "nonce": session_nonces.get(wid_str),
            }

            env = payload  # no envelope, no signatures
            task_out.send_multipart([
                wid_str.encode(),
                json.dumps(env, separators=(",", ":"), sort_keys=True).encode(),
            ])
            num_tasks += 1

        # Receive results
        grads_accum = []
        losses = []
        counts = []
        for _ in range(num_tasks):
            result = results_in.recv_json()
            wid = result.get("worker_id")
            wid_str = str(wid)

            if not isinstance(result, dict):
                print(f"Coordinator: malformed result from worker {wid}", flush=True)
                continue

            # Check nonce matches what we assigned to that worker
            expected_nonce = session_nonces.get(wid_str)
            res_nonce = result.get("nonce")
            if expected_nonce is None:
                print(f"Coordinator: no session nonce for worker {wid_str}; ignoring result", flush=True)
                continue

            if res_nonce != expected_nonce:
                print(
                    f"Coordinator: nonce mismatch from worker {wid_str}: "
                    f"expected={expected_nonce}, got={res_nonce}; ignoring result.",
                    flush=True,
                )
                continue

            grads_accum.append(result["grads"])  # dict[name -> list]
            losses.append(result["loss"])
            counts.append(result["n"])

        if not grads_accum:
            print(f"Coordinator: no valid gradients received at epoch {epoch}; skipping update.", flush=True)
            continue

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
            flush=True,
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
        f"\nTiming summary: handshake={handshake_time:.3f}s | preprocess={preprocess_time:.3f}s | "
        f"total_training={total_training_time:.3f}s | avg_epoch={avg_epoch:.3f}s",
        flush=True,
    )

    # Send SHUTDOWN control message to all workers so they can exit cleanly.
    try:
        shutdown_payload_base = {"control": "SHUTDOWN"}
        for wid in range(cfg.NUM_WORKERS):
            wid_str = str(wid)
            payload = dict(shutdown_payload_base)
            payload["worker_id"] = wid_str
            payload["nonce"] = session_nonces.get(wid_str)
            task_out.send_multipart([
                wid_str.encode(),
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(),
            ])
        print("Coordinator: sent SHUTDOWN to all workers", flush=True)
    except Exception:
        print("Coordinator: failed to send SHUTDOWN to workers", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-json", type=str, default=None,
                   help="Optional path to a model JSON file describing architecture")
    p.add_argument("--initial-weights", type=str, default=None,
                   help="Optional path to a JSON file with initial weights {param_name: nested lists}")
    args = p.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    cfg = CoordinatorConfig()
    cfg.NUM_WORKERS = args.num_workers
    cfg.EPOCHS = args.epochs
    cfg.LR = args.lr
    cfg.SEED = seed
    cfg.MODEL_JSON = args.model_json
    cfg.INITIAL_WEIGHTS = args.initial_weights

    train(cfg)


if __name__ == "__main__":
    main()
