
import os
import binascii


class WorkerConfig:

    WORKER_ID = int(os.environ.get("WORKER_ID", 0))

    TASK_ENDPOINT = "tcp://coordinator:5557"
    RESULT_ENDPOINT = "tcp://coordinator:5558"
    HANDSHAKE_ENDPOINT = "tcp://coordinator:5560"

    PRIVATE_KEY_PATH = os.path.join("keys", f"worker{WORKER_ID}_private.pem")
    COORDINATOR_PUBLIC_KEY_PATH = os.path.join("keys", "coordinator_public.pem")

