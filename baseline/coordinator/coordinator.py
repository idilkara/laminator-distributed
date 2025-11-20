import argparse
import time
import os, random
import json
import numpy as np
import torch
import torch.nn as nn  # noqa: F401 (kept if your models import needs it)
import zmq

from models import ModelHandler
from data_preprocess import process_census
from training import batches_for_workers, _avg_grads_and_loss

from config import CoordinatorConfig


def train(cfg: CoordinatorConfig):
    ctx = zmq.Context.instance()

    # ---- data sockets ----
    # PUSH tasks to workers
    task_out = ctx.socket(zmq.PUSH)
    task_out.bind(cfg.TASK_ENDPOINT)

    # PULL results from workers
    results_in = ctx.socket(zmq.PULL)
    results_in.bind(cfg.RESULT_ENDPOINT)

    print("Coordinator started (no handshake / no security).", flush=True)

    # ----- Data preparation -----
    preprocess_start = time.time()
    X_train, y_train, X_test, y_test = process_census()
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start
    print(f"Coordinator: data preprocessing completed in {preprocess_time:.3f}s", flush=True)
    rng = np.random.default_rng(cfg.SEED)

    # Initialize model (JSON). Allow overriding via cfg.MODEL_JSON file path.
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

    if getattr(cfg, "MODEL_JSON", None):
        try:
            with open(cfg.MODEL_JSON, "r") as mf:
                model_string = mf.read()
        except Exception as e:
            print(f"Coordinator: failed to read model JSON {cfg.MODEL_JSON}: {e}; falling back to default", flush=True)
            model_string = model_string
    else:
        model_string = model_string

    model = ModelHandler.parse_model_string(model_string)  # ensure same as workers
    assert model != -1

    # Optional: load initial weights if provided (this is not security-related)
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
        f"Coordinator started with {cfg.NUM_WORKERS} workers. "
        f"Data: X={X_train.shape}, y={y_train.shape}",
        flush=True
    )

    # ---------- Training epochs ----------
    training_start = time.time()
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        # Send one task per logical worker (PUSH will load-balance across actual workers)
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
            }
            task_out.send_json(payload)
            num_tasks += 1

        # Receive results for all tasks we sent
        grads_accum = []
        losses = []
        counts = []
        for _ in range(num_tasks):
            result = results_in.recv_json()
            wid = result.get("worker_id")
            if not isinstance(result, dict):
                print(f"Coordinator: malformed result from worker {wid}", flush=True)
                continue

            grads_accum.append(result["grads"])  # dict[name -> list]
            losses.append(result["loss"])
            counts.append(result["n"])

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
            logits = model(X_test_t)  # shape: [N, 2]
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
        f"\nTiming summary: preprocess={preprocess_time:.3f}s | "
        f"total_training={total_training_time:.3f}s | avg_epoch={avg_epoch:.3f}s",
        flush=True,
    )

    # Send SHUTDOWN control messages (one per worker; PUSH load-balances)
    try:
        shutdown_payload = {"control": "SHUTDOWN"}
        for _ in range(cfg.NUM_WORKERS):
            task_out.send_json(shutdown_payload)
        print("Coordinator: sent SHUTDOWN to workers", flush=True)
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
