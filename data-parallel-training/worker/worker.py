# worker.py
import random
import json
import time
import zmq
import numpy as np
from training import compute_grad_and_loss
from auth import verify_handshake_offer, create_handshake_response, sign_envelope, verify_envelope
from config import WorkerConfig
import sys
import hashlib


def main():
    ctx = zmq.Context.instance()

    # ---- Training sockets ----
    receiver = ctx.socket(zmq.DEALER)
    receiver.setsockopt(zmq.IDENTITY, str(WorkerConfig.WORKER_ID).encode())
    receiver.connect(WorkerConfig.TASK_ENDPOINT)

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

    # Step 3: Verify coordinator’s signature
    coord_pub = WorkerConfig.COORDINATOR_PUBLIC_KEY_PATH
    if not verify_handshake_offer(offer, coord_pub):
        print(f"Worker {WorkerConfig.WORKER_ID}: invalid handshake offer signature", flush=True)
        sys.exit(1)

    # Step 4: Create and send signed response via PUSH → PULL
    response = create_handshake_response(
        offer, str(WorkerConfig.WORKER_ID), WorkerConfig.PRIVATE_KEY_PATH
    )
    out = {"wid": WorkerConfig.WORKER_ID, "handshake_response": response}
    sender.send_json(out)
    print(f"Worker {WorkerConfig.WORKER_ID}: sent handshake response to coordinator", flush=True)

    # Save session nonce from offer for later message signing/verification
    session_nonce = offer.get("nonce")
    print(f"Worker {WorkerConfig.WORKER_ID}: saved session_id={offer.get('session_id')} nonce={session_nonce}", flush=True)
    handshake_end = time.time()
    handshake_time = handshake_end - handshake_start

    # Step 5: Continue normal operation
    seen_nonces_tasks = set()

    # --- Timing accumulators ---
    gradloss_total_time = 0.0
    gradloss_count = 0

    total_time_for_verifying_envelopes = 0.0
    total_time_for_hashing = 0.0
    total_time_for_signing_envelopes = 0.0
    # ---- Main loop ----
    while True:
        # Receive and verify the task envelope
        env = receiver.recv_json()

        # Expect envelope like: {"wid": <wid>, "payload": {"message": <json str>, "signature": <hex str>}}
        payload_env = env.get("payload")
        if not isinstance(payload_env, dict) or "message" not in payload_env or "signature" not in payload_env:
            print(f"Worker {WorkerConfig.WORKER_ID}: malformed envelope from coordinator", flush=True)
            continue

        t_verify_start = time.time()
        # Verify envelope using stored session nonce
        coord_pub = WorkerConfig.COORDINATOR_PUBLIC_KEY_PATH
        if not session_nonce:
            print(f"Worker {WorkerConfig.WORKER_ID}: no session nonce; ignoring inbound task", flush=True)
            continue
        if not verify_envelope(payload_env, session_nonce, coord_pub):
            print(f"Worker {WorkerConfig.WORKER_ID}: envelope verification failed; ignoring", flush=True)
            continue

        t_verify_end = time.time()
        total_time_for_verifying_envelopes += (t_verify_end - t_verify_start)

        # Extract task payload and remove nonce
        task = dict(payload_env["message"]) if isinstance(payload_env["message"], dict) else {}
        task.pop("nonce", None)

        # Handle control/shutdown messages from coordinator.
        is_shutdown = False
        try:
            if isinstance(task, dict):
                if task.get("control") == "SHUTDOWN" :
                    is_shutdown = True
                if task.get("shutdown") is True:
                    is_shutdown = True
        except Exception:
            is_shutdown = False

        if is_shutdown:
            avg_gradloss = (gradloss_total_time / gradloss_count) if gradloss_count > 0 else 0.0
            avg_hashing = (total_time_for_hashing / gradloss_count) if gradloss_count > 0 else 0.0
            avg_signing = (total_time_for_signing_envelopes / gradloss_count) if gradloss_count > 0 else 0.0
            avg_verifying = (total_time_for_verifying_envelopes / gradloss_count) if gradloss_count > 0 else 0.0



            print(f"Worker {WorkerConfig.WORKER_ID}: handshake time: {handshake_time:.6f}s", flush=True)
            print(f"Worker {WorkerConfig.WORKER_ID}: average gradient loss compute time: {avg_gradloss:.6f}s over {gradloss_count} tasks", flush=True)
            print(f"Worker {WorkerConfig.WORKER_ID}: average hashing time: {avg_hashing:.6f}s over {gradloss_count} tasks", flush=True)
            print(f"Worker {WorkerConfig.WORKER_ID}: average envelope signing time: {avg_signing:.6f}s over {gradloss_count} tasks", flush=True)
            print(f"Worker {WorkerConfig.WORKER_ID}: average envelope verifying time: {avg_verifying:.6f}s over {gradloss_count} tasks", flush=True)

            print(f"Worker {WorkerConfig.WORKER_ID}: received SHUTDOWN from coordinator; exiting.", flush=True)
            try:
                receiver.close(linger=0)
                sender.close(linger=0)
                ctx.term()
            except Exception:
                pass
            sys.exit(0)

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

        t_hash_start = time.time()

        # Compute required hashes per spec:
        # H(DTr_i) -> hash of training data (X,y)
        # H(MAr) -> hash of model architecture
        # H(Me_init) -> hash of initial model weights (as sent in task)
        # H(T) -> hash of training configuration (lr, epoch, etc.)
        def _stable_json_hash(obj):
            try:
                j = json.dumps(obj, sort_keys=True, separators=(",",":"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            except Exception:
                j = json.dumps(str(obj), sort_keys=True, separators=(",",":"))
            return hashlib.sha256(j.encode()).hexdigest()

        H_DTr = _stable_json_hash({"X": task.get("X"), "y": task.get("y")})
        H_MAr = _stable_json_hash(task.get("architecture"))
        H_Me_init = _stable_json_hash(task.get("weights"))
        H_T = _stable_json_hash({"lr": task.get("lr"), "epoch": task.get("epoch")})
  
        t_hash_end = time.time()
        total_time_for_hashing += (t_hash_end - t_hash_start)


        ## CORRECTNESS TEST: WRONG HASH
        # old = H_Me_init
        # H_Me_init = ("0" * 64)
        # print(f"Worker {WorkerConfig.WORKER_ID}: intentionally corrupting H_Me_init (was {old[:12]}...) -> {H_Me_init[:12]}...; remaining misbehave={misbehave_left}", flush=True)

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
            # New fields per requested message format
            "hashes": {
                "H_DTr": H_DTr,
                "H_MAr": H_MAr,
                "H_Me_init": H_Me_init,
                "H_T": H_T,
            },
            "epoch": int(task.get("epoch", -1)), 
            "worker_index": WorkerConfig.WORKER_ID,
            "trained_weights": trained_weights_json,
        }

        # ## CORRECTNESS TEST: WRONG EPOCH 
        # if int(task.get("epoch", -1)) == 2:
        #     old_epoch = payload["epoch"]
        #     payload["epoch"] = 99
        #     print(f"Worker {WorkerConfig.WORKER_ID}: intentionally corrupting epoch (was {old_epoch}) -> {payload['epoch']}", flush=True)

        t_sign_start = time.time()
        # Sign and send back the result using the established session nonce
        if not session_nonce:
            print(f"Worker {WorkerConfig.WORKER_ID}: no session nonce when sending results; dropping", flush=True)
            continue
        result_envelope = sign_envelope(payload, session_nonce, WorkerConfig.PRIVATE_KEY_PATH)

        # # CORRECTNESS TEST: CHANGE SIGNATURE TO INVALID
        # old_sig = result_envelope.get("signature")
        # if old_sig:
        #     result_envelope["signature"] = "00" * (len(old_sig) // 2)
        #     print(f"Worker {WorkerConfig.WORKER_ID}: intentionally corrupting signature", flush=True)

        t_sign_end = time.time()
        total_time_for_signing_envelopes += (t_sign_end - t_sign_start)

        out_env = {"wid": WorkerConfig.WORKER_ID, "payload": result_envelope}
        sender.send_json(out_env)
        print(f"Worker {WorkerConfig.WORKER_ID}: sent results to coordinator.", flush=True)


if __name__ == "__main__":
    main()
