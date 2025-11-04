# coordinator.py
import argparse
import time
from dataclasses import dataclass
import os, random
import numpy as np
import torch
import torch.nn as nn  # noqa: F401 (kept if your models import needs it)
import zmq

from models import ModelHandler
from data_preprocess import process_census
from training import batches_for_workers

from auth import coordinator_handle_handshake, sign_envelope, verify_envelope

HANDSHAKE_ENDPOINT = "tcp://*:5556"
TASK_ENDPOINT      = "tcp://*:5557"
RESULT_ENDPOINT    = "tcp://*:5558"


@dataclass
class Config:
    num_workers: int
    epochs: int = 15
    lr: float = 0.1
    seed: int = 42


def _avg_grads_and_loss(grads_list, counts_list, losses_list):
    """
    grads_list: list[dict[param_name -> list[float]]]  (JSON from workers)
    counts_list: list[int]  (per-worker number of samples)
    losses_list: list[float]

    Returns:
      avg_grads: dict[param_name -> torch.Tensor]
      avg_loss: float (weighted by counts)
    """
    if not grads_list:
        return {}, 0.0

    # Union of param names across workers
    param_names = set()
    for gd in grads_list:
        param_names.update(gd.keys())

    total_n = float(sum(int(n) for n in counts_list))
    if total_n <= 0:
        total_n = 1.0

    # Weighted average of grads by sample count
    avg_grads = {}
    for name in sorted(param_names):
        acc = None
        for gd, n in zip(grads_list, counts_list):
            if name not in gd:
                continue
            g = torch.tensor(gd[name], dtype=torch.float32)
            weight = float(n) / total_n
            acc = g * weight if acc is None else acc + g * weight
        if acc is None:
            # If no worker provided this param (shouldn't happen), set zeros
            acc = torch.tensor(0.0)
        avg_grads[name] = acc

    # Weighted average loss
    wloss = 0.0
    for n, l in zip(counts_list, losses_list):
        wloss += float(l) * (float(n) / total_n)
    return avg_grads, float(wloss)


# ---------- Training Loop ----------
def train(cfg: Config):
    ctx = zmq.Context.instance()

    # ---- handshake server (REP) ----
    hs_rep = ctx.socket(zmq.REP); hs_rep.bind(HANDSHAKE_ENDPOINT)
    session_keys = {}  # wid -> bytes

    # ---- data sockets ----
    # Use ROUTER so we can address tasks to specific workers (by identity).
    task_out = ctx.socket(zmq.ROUTER);  task_out.bind(TASK_ENDPOINT)
    # Results can still be collected via PULL from workers' PUSH sockets
    results_in = ctx.socket(zmq.PULL); results_in.bind(RESULT_ENDPOINT)

    prov_secret = os.environ.get("HMAC_PROVISIONING_SECRET", "").encode()
    if len(prov_secret) < 16:
        raise RuntimeError("Set strong HMAC_PROVISIONING_SECRET in the environment")
    print("Coordinator started with handshake server.", flush=True)

    # Data
    X_train, y_train, X_test, y_test = process_census()
    rng = np.random.default_rng(cfg.seed)

    # Initialize model (exactly as your JSON)
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
        f"Coordinator started with {cfg.num_workers} workers. "
        f"Data: X={X_train.shape}, y={y_train.shape}",
        flush=True
    )

    print(f"Coordinator up. Waiting for {cfg.num_workers} workers to handshake…", flush=True)
    seen_nonces_results = set()

    # --- Accept N successful handshakes ---
    while len(session_keys) < cfg.num_workers:
        msg = hs_rep.recv_json()
        reply, ready = coordinator_handle_handshake(msg, prov_secret)
        hs_rep.send_json(reply)
        if ready:
            wid, sk = ready
            session_keys[str(wid)] = sk
            print(f"Session established with worker {wid}", flush=True)

    print("All workers authenticated. Starting training.", flush=True)


    partition_start_idx = 0
    partition_size = X_train.shape[0] // cfg.epochs
    print(f"Each epoch will use data partition of size {partition_size}", flush=True)

    
    # ---------- Training epochs ----------
    for epoch in range(cfg.epochs):
        t0 = time.time()

        # Send tasks securely
        for wid, Xb, yb in batches_for_workers(X_train[partition_start_idx:partition_start_idx + partition_size], y_train[partition_start_idx:partition_start_idx + partition_size], cfg.num_workers, rng):
            wid_str = str(wid)
            payload = {
                "worker_id": wid_str,
                "X": Xb.tolist(),
                "y": yb.tolist(),
                # Workers expect 'architecture' and 'weights'
                "architecture": model_string,
                "weights": {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
                "lr": float(cfg.lr),
                "epoch": int(epoch)
            }
            env = sign_envelope(session_keys[wid_str], wid_str, payload)
            # Router expects [identity, payload]; send the JSON payload as a
            # single frame. Worker DEALER socket (with identity set) will
            # receive only the payload frame.
            import json as _json
            task_out.send_multipart([wid_str.encode(), _json.dumps(env, separators=(",",":"), sort_keys=True).encode()])
        partition_start_idx += partition_size
        # Receive results securely
        grads_accum = []
        losses = []
        counts = []
        for _ in range(cfg.num_workers):
            env = results_in.recv_json()
            wid = env["wid"]
            payload = verify_envelope(env, session_keys[wid], seen_nonces=seen_nonces_results)

            grads_accum.append(payload["grads"])  # dict[name -> list]
            losses.append(payload["loss"])
            counts.append(payload["n"])

        # Average and apply gradients
        avg_grads, avg_loss = _avg_grads_and_loss(grads_accum, counts, losses)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in avg_grads:
                    continue
                g = avg_grads[name].to(param.device, dtype=param.dtype)
                param -= cfg.lr * g

        dt = time.time() - t0

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_test_t)                     # shape: [N, 2]
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        acc = float(np.mean(y_pred == y_test))

        print(
            f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f} | time={dt:.2f}s | Acc={acc:.4f}",
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

    cfg = Config(num_workers=args.num_workers, epochs=args.epochs, lr=args.lr, seed=seed)
    train(cfg)


if __name__ == "__main__":
    main()
