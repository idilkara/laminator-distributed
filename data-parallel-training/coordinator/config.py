
import os
import binascii


## coordinator verifies worker signatures using their public keys


class CoordinatorConfig:

    TASK_ENDPOINT = "tcp://*:5557"
    RESULT_ENDPOINT = "tcp://*:5558"
    HANDSHAKE_ENDPOINT = "tcp://*:5560"

    PRIVATE_KEY_PATH = os.path.join("keys", "coordinator_private.pem")

    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))

    WORKER_KEY_PATHS = {}
    for wid in range(NUM_WORKERS):
        WORKER_KEY_PATHS[wid] = os.path.join("keys", f"worker{wid}_public.pem")

