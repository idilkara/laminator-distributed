# worker.py
import os
import time
import random
import json
import torch
import zmq
import numpy as np
from training import compute_grad_and_loss
from auth import sign_message, verify_signature
from config import WorkerConfig
import sys




def main():
    ctx = zmq.Context.instance()



    # ---- Training sockets ----
    # Use DEALER and set our identity so the coordinator (ROUTER) can
    # address tasks specifically to this worker.
    receiver = ctx.socket(zmq.DEALER)
    receiver.setsockopt(zmq.IDENTITY, str(WorkerConfig.WORKER_ID).encode())
    receiver.connect(WorkerConfig.TASK_ENDPOINT)
    sender = ctx.socket(zmq.PUSH)
    sender.connect(WorkerConfig.RESULT_ENDPOINT)

    seen_nonces_tasks = set()
    time.sleep(1)

    # ---- Main loop ----
    while True:
        # Receive and verify the task envelope
        env = receiver.recv_json()

        # Expect envelope like: {"wid": <wid>, "payload": {"message": <json str>, "signature": <hex str>}}
        payload_env = env.get("payload")
        if not isinstance(payload_env, dict) or "message" not in payload_env or "signature" not in payload_env:
            print(f"Worker {WorkerConfig.WORKER_ID}: malformed envelope from coordinator", flush=True)
            continue

        message_json = payload_env["message"]
        sig_hex = payload_env["signature"]
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except Exception:
            print(f"Worker {WorkerConfig.WORKER_ID}: invalid signature encoding", flush=True)
            continue

        # Verify signature using coordinator's public key
        coord_pub = WorkerConfig.COORDINATOR_PUBLIC_KEY_PATH
        if not verify_signature(message_json.encode(), sig_bytes, coord_pub):
            print(f"Worker {WorkerConfig.WORKER_ID}: signature verification failed for incoming task; ignoring", flush=True)
            continue

        # Parse inner task message
        try:
            task = json.loads(message_json)
        except Exception:
            print(f"Worker {WorkerConfig.WORKER_ID}: failed to parse task JSON", flush=True)
            continue
        # Handle control/shutdown messages from coordinator.
        # Accept several common forms so coordinator can send a simple
        # control envelope like {"control": "SHUTDOWN"} or a bare
        # string "SHUTDOWN".
        is_shutdown = False
        try:
            if isinstance(task, dict):
                # common keys that might indicate shutdown
                if task.get("control") == "SHUTDOWN" or task.get("command") == "SHUTDOWN" or task.get("type") == "SHUTDOWN":
                    is_shutdown = True
                # also allow explicit boolean flag
                if task.get("shutdown") is True:
                    is_shutdown = True
            else:
                # message could be a plain string
                if isinstance(task, str) and task.upper() == "SHUTDOWN":
                    is_shutdown = True
        except Exception:
            is_shutdown = False

        if is_shutdown:
            print(f"Worker {WorkerConfig.WORKER_ID}: received SHUTDOWN from coordinator; exiting.", flush=True)
            try:
                # Close sockets and terminate context cleanly
                receiver.close(linger=0)
                sender.close(linger=0)
                ctx.term()
            except Exception:
                pass
            sys.exit(0)
        print(f"Worker {WorkerConfig.WORKER_ID}: received task for epoch {task['epoch']}", flush=True)
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
            "worker_id": WorkerConfig.WORKER_ID,
            "grads": grads_json,
            "loss": float(loss),
            "n": int(n),
        }

        # Sign and send back the result. We send an envelope with message + signature (hex)
        result_msg = json.dumps(payload, separators=(",",":"), sort_keys=True)
        sig = sign_message(result_msg.encode(), WorkerConfig.PRIVATE_KEY_PATH)
        result_envelope = {"message": result_msg, "signature": sig.hex()}
        out_env = {"wid": WorkerConfig.WORKER_ID, "payload": result_envelope}
        sender.send_json(out_env)
        print(f"Worker {WorkerConfig.WORKER_ID}: sent results to coordinator.", flush=True)


if __name__ == "__main__":
    main()
