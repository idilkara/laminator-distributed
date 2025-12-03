# baseline_worker_router.py
import random
import json
import time
import zmq
import numpy as np
from training import compute_grad_and_loss
from config import WorkerConfig
import sys


def main():
    ctx = zmq.Context.instance()

    # ---- Training sockets ----
    # ROUTER <-> DEALER for tasks
    receiver = ctx.socket(zmq.DEALER)
    receiver.setsockopt(zmq.IDENTITY, str(WorkerConfig.WORKER_ID).encode())
    receiver.connect(WorkerConfig.TASK_ENDPOINT)

    # PUSH -> PULL for results
    sender = ctx.socket(zmq.PUSH)
    sender.connect(WorkerConfig.RESULT_ENDPOINT)

    # ---- Handshake ----
    handshake_start = time.time()
    # Step 1: Send HELLO so coordinator’s ROUTER learns our identity
    hello = {"type": "hello", "wid": WorkerConfig.WORKER_ID}
    receiver.send_json(hello)
    print(f"Worker {WorkerConfig.WORKER_ID}: sent HELLO to coordinator", flush=True)

    # Step 2: Wait for handshake offer from coordinator (ROUTER → DEALER)
    try:
        offer_env = receiver.recv_json()
    except Exception as e:
        print(f"Worker {WorkerConfig.WORKER_ID}: failed to receive handshake offer ({e})", flush=True)
        sys.exit(1)

    offer = offer_env.get("handshake")
    if not offer:
        print(f"Worker {WorkerConfig.WORKER_ID}: no handshake offer received", flush=True)
        sys.exit(1)

    # In baseline, we TRUST the offer (no signatures), just extract session_id + nonce
    session_id = offer.get("session_id")
    session_nonce = offer.get("nonce")
    if not session_id or not session_nonce:
        print(f"Worker {WorkerConfig.WORKER_ID}: invalid handshake offer (missing session_id or nonce)", flush=True)
        sys.exit(1)

    print(
        f"Worker {WorkerConfig.WORKER_ID}: received handshake offer "
        f"session_id={session_id} nonce={session_nonce}",
        flush=True,
    )

    # Step 3: Create and send a simple (unsigned) handshake response via PUSH → PULL
    response = {
        "wid": WorkerConfig.WORKER_ID,
        "session_id": session_id,
        "nonce": session_nonce,
        "ok": True,
    }
    out = {"wid": WorkerConfig.WORKER_ID, "handshake_response": response}
    sender.send_json(out)
    print(f"Worker {WorkerConfig.WORKER_ID}: sent handshake response to coordinator", flush=True)

    handshake_end = time.time()
    handshake_time = handshake_end - handshake_start

    # Step 4: Continue normal operation
    # --- Timing accumulators ---
    gradloss_total_time = 0.0
    gradloss_count = 0

    while True:
        # Receive task/control message directly (no envelope/signature)
        task = receiver.recv_json()

        # Handle control/shutdown messages from coordinator.
        is_shutdown = False
        try:
            if isinstance(task, dict):
                if task.get("control") == "SHUTDOWN":
                    is_shutdown = True
                if task.get("shutdown") is True:
                    is_shutdown = True
        except Exception:
            is_shutdown = False

        if is_shutdown:
            avg_gradloss = (gradloss_total_time / gradloss_count) if gradloss_count > 0 else 0.0
            print(f"Worker {WorkerConfig.WORKER_ID}: handshake time: {handshake_time:.6f}s", flush=True)
            print(
                f"Worker {WorkerConfig.WORKER_ID}: average gradient loss compute time: "
                f"{avg_gradloss:.6f}s over {gradloss_count} tasks",
                flush=True,
            )
            print(f"Worker {WorkerConfig.WORKER_ID}: received SHUTDOWN from coordinator; exiting.", flush=True)
            try:
                receiver.close(linger=0)
                sender.close(linger=0)
                ctx.term()
            except Exception:
                pass
            sys.exit(0)

        # Basic nonce check (baseline, no crypto)
        incoming_nonce = task.get("nonce")
        if incoming_nonce != session_nonce:
            print(
                f"Worker {WorkerConfig.WORKER_ID}: WARNING: task nonce mismatch "
                f"(expected={session_nonce}, got={incoming_nonce}); processing anyway (baseline).",
                flush=True,
            )

        # RECEIVED A TRAINING TASK:
        print(f"Worker {WorkerConfig.WORKER_ID}: received task for epoch {task['epoch']}", flush=True)
        print(len(task['X']), "samples")

        # --- Timing: grad/loss compute ---
        t_comp_start = time.time()
        grads, loss, n, updated_state = compute_grad_and_loss(task)
        t_comp_end = time.time()
        gradloss_time = t_comp_end - t_comp_start
        gradloss_total_time += gradloss_time
        gradloss_count += 1

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

        # Convert updated_state (trained weights) into JSON-serializable form
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

        payload = {
            "worker_id": WorkerConfig.WORKER_ID,
            "grads": grads_json,
            "loss": float(loss),
            "n": int(n),
            "epoch": int(task.get("epoch", -1)),
            "worker_index": WorkerConfig.WORKER_ID,
            "trained_weights": trained_weights_json,
            # Echo back the session nonce
            "nonce": session_nonce,
        }

        sender.send_json(payload)
        print(
            f"Worker {WorkerConfig.WORKER_ID}: sent results to coordinator "
            f"(epoch={payload['epoch']}, nonce={session_nonce}).",
            flush=True,
        )


if __name__ == "__main__":
    main()
