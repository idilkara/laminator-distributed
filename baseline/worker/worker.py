import time
import random
import json
import torch
import zmq
import numpy as np
from training import compute_grad_and_loss
from config import WorkerConfig
import sys



def main():
    ctx = zmq.Context.instance()

    # ---- Training sockets ----
    # PULL tasks from coordinator
    receiver = ctx.socket(zmq.PULL)
    receiver.connect(WorkerConfig.TASK_ENDPOINT)

    # PUSH results back
    sender = ctx.socket(zmq.PUSH)
    sender.connect(WorkerConfig.RESULT_ENDPOINT)

    print(f"Worker {WorkerConfig.WORKER_ID}: started (no handshake / no security).", flush=True)

    while True:
        # Receive task or control message
        task = receiver.recv_json()

        # Handle control/shutdown messages from coordinator.
        is_shutdown = False
        try:
            if isinstance(task, dict):
                if (
                    task.get("control") == "SHUTDOWN"
                    or task.get("command") == "SHUTDOWN"
                    or task.get("type") == "SHUTDOWN"
                ):
                    is_shutdown = True
                if task.get("shutdown") is True:
                    is_shutdown = True
            else:
                if isinstance(task, str) and task.upper() == "SHUTDOWN":
                    is_shutdown = True
        except Exception:
            is_shutdown = False

        if is_shutdown:
            print(f"Worker {WorkerConfig.WORKER_ID}: received SHUTDOWN; exiting.", flush=True)
            try:
                receiver.close(linger=0)
                sender.close(linger=0)
                ctx.term()
            except Exception:
                pass
            sys.exit(0)

        # Normal training task
        print(f"Worker {WorkerConfig.WORKER_ID}: received task for epoch {task['epoch']}", flush=True)
        print(len(task["X"]), "samples")

        # Compute gradients, loss and obtain updated (trained) weights
        grads, loss, n, updated_state = compute_grad_and_loss(task)
        

        # Convert gradients to JSON-serializable format
        grads_json = {}
        for k, v in grads.items():
            try:
                if hasattr(v, "cpu"):
                    grads_json[k] = v.cpu().tolist()
                elif hasattr(v, "tolist"):
                    grads_json[k] = v.tolist()
                else:
                    grads_json[k] = list(v)
            except Exception:
                try:
                    grads_json[k] = [float(x) for x in v]
                except Exception:
                    grads_json[k] = str(v)

        # Optionally convert trained weights to JSON-serializable form
        trained_weights_json = {}
        for k, v in updated_state.items():
            try:
                if hasattr(v, "tolist"):
                    trained_weights_json[k] = v.tolist()
                else:
                    trained_weights_json[k] = list(v)
            except Exception:
                try:
                    trained_weights_json[k] = [float(x) for x in v]
                except Exception:
                    trained_weights_json[k] = str(v)

        result = {
            "worker_id": WorkerConfig.WORKER_ID,
            "grads": grads_json,
            "loss": float(loss),
            "n": int(n),
            "epoch": int(task.get("epoch", -1)),
            "worker_index": WorkerConfig.WORKER_ID,
            "trained_weights": trained_weights_json,
        }

        sender.send_json(result)
        print(f"Worker {WorkerConfig.WORKER_ID}: sent results to coordinator.", flush=True)


if __name__ == "__main__":
    main()
