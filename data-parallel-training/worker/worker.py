# worker.py
import os
import time
import random
import torch
import zmq
import numpy as np
from training import compute_grad_and_loss
from auth import worker_start_handshake, sign_envelope, verify_envelope

HANDSHAKE_ENDPOINT = "tcp://coordinator:5556"
TASK_ENDPOINT      = "tcp://coordinator:5557"
RESULT_ENDPOINT    = "tcp://coordinator:5558"

WORKER_ID = os.environ.get("WORKER_ID", "0")  # must be unique per worker


def main():
    ctx = zmq.Context.instance()

    # ---- Handshake (REQ/REP) ----
    hs_req = ctx.socket(zmq.REQ)
    hs_req.connect(HANDSHAKE_ENDPOINT)

    def _send_req(obj): hs_req.send_json(obj)
    def _recv_rep():    return hs_req.recv_json()

    # provisioning secret must exist on worker
    prov_secret = os.environ.get("HMAC_PROVISIONING_SECRET", "")
    if len(prov_secret) < 16:
        raise RuntimeError("Set HMAC_PROVISIONING_SECRET on worker (≥16 bytes)")

    print(f"Worker {WORKER_ID}: starting handshake with coordinator...", flush=True)
    session_key = worker_start_handshake(_send_req, _recv_rep, WORKER_ID)
    print(f"Worker {WORKER_ID}: handshake complete. Session key established.", flush=True)

    # ---- Training sockets ----
    # Use DEALER and set our identity so the coordinator (ROUTER) can
    # address tasks specifically to this worker.
    receiver = ctx.socket(zmq.DEALER)
    receiver.setsockopt(zmq.IDENTITY, WORKER_ID.encode())
    receiver.connect(TASK_ENDPOINT)
    sender = ctx.socket(zmq.PUSH)
    sender.connect(RESULT_ENDPOINT)

    seen_nonces_tasks = set()
    time.sleep(1)

    # ---- Main loop ----
    while True:
        # Receive and verify the task envelope
        env = receiver.recv_json()
        task = verify_envelope(env, session_key, seen_nonces=seen_nonces_tasks)
        print(f"Worker {WORKER_ID}: received task for epoch {task['epoch']}", flush=True)
        print(len(task['X']), "samples")
        # Compute gradients and loss
        grads, loss, n = compute_grad_and_loss(task)
        time.sleep(random.uniform(0.05, 0.2))  # simulate variable workload

        # Convert gradients to JSON-serializable format. `grads` may contain
        # torch tensors or numpy arrays depending on where they were created;
        # handle both cases robustly.
        grads_json = {}
        for k, v in grads.items():
            try:
                # torch.Tensor has .cpu()
                if hasattr(v, "cpu"):
                    grads_json[k] = v.cpu().tolist()
                elif hasattr(v, "tolist"):
                    grads_json[k] = v.tolist()
                else:
                    grads_json[k] = list(v)
            except Exception:
                # Fallback: coerce to list via Python iteration
                try:
                    grads_json[k] = [float(x) for x in v]
                except Exception:
                    grads_json[k] = str(v)

        payload = {
            "worker_id": WORKER_ID,
            "grads": grads_json,
            "loss": float(loss),
            "n": int(n),
        }

        # Sign and send back the result
        env_out = sign_envelope(session_key, WORKER_ID, payload)
        sender.send_json(env_out)
        print(f"Worker {WORKER_ID}: sent results to coordinator.", flush=True)


if __name__ == "__main__":
    main()
