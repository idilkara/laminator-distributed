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

from models import ModelHandler
from data_preprocess import process_census
from training import batches_for_workers, _avg_grads_and_loss

from auth import sign_message, verify_signature

# Load CoordinatorConfig to read default endpoints and key paths
from config import CoordinatorConfig


# ---------- Training Loop ----------
def train(cfg: CoordinatorConfig):
    ctx = zmq.Context.instance()

    # ---- handshake server (REP) ----
    hs_rep = ctx.socket(zmq.REP); hs_rep.bind(cfg.HANDSHAKE_ENDPOINT)
    session_keys = {}  # wid -> bytes

    # ---- data sockets ----
    # Use ROUTER so we can address tasks to specific workers (by identity).
    task_out = ctx.socket(zmq.ROUTER);  task_out.bind(cfg.TASK_ENDPOINT)
    # Results can still be collected via PULL from workers' PUSH sockets
    results_in = ctx.socket(zmq.PULL); results_in.bind(cfg.RESULT_ENDPOINT)

    # Using RSA-based handshake (keys in ./keys/*.pem)
    print("Coordinator started with RSA handshake server.", flush=True)

    # Data preparation
    X_train, y_train, X_test, y_test = process_census()
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

    
    # ---------- Training epochs ----------
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

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

            # Serialize and sign the payload, then send an envelope {message, signature}
            msg_str = json.dumps(payload, separators=(",",":"), sort_keys=True)
            sig = sign_message(msg_str.encode(), cfg.PRIVATE_KEY_PATH)
            signed_envelope = {"message": msg_str, "signature": sig.hex()}

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
                continue

            msg_str = payload_env["message"]
            try:
                sig_bytes = bytes.fromhex(payload_env["signature"])
            except Exception:
                print(f"Coordinator: invalid signature encoding from worker {wid}", flush=True)
                continue

            # Look up worker public key and verify
            try:
                worker_pub = cfg.WORKER_KEY_PATHS[int(wid)]
            except Exception:
                print(f"Coordinator: no public key for worker {wid}", flush=True)
                continue

            if not verify_signature(msg_str.encode(), sig_bytes, worker_pub):
                print(f"Coordinator: signature verification failed for worker {wid}; ignoring", flush=True)
                continue

            # Parse the worker's inner message
            try:
                worker_payload = json.loads(msg_str)
            except Exception:
                print(f"Coordinator: failed to parse worker {wid} message JSON", flush=True)
                continue

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
    # Send SHUTDOWN control message to all workers so they can exit cleanly.
    try:
        shutdown_payload = {"control": "SHUTDOWN"}
        msg_str = json.dumps(shutdown_payload, separators=(",",":"), sort_keys=True)
        sig = sign_message(msg_str.encode(), cfg.PRIVATE_KEY_PATH)
        signed_envelope = {"message": msg_str, "signature": sig.hex()}
        for wid in range(cfg.NUM_WORKERS):
            wid_str = str(wid)
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
