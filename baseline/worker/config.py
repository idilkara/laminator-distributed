
import os
import binascii


class WorkerConfig:

    WORKER_ID = int(os.environ.get("WORKER_ID", 0))

    TASK_ENDPOINT = "tcp://coordinator:5557"
    RESULT_ENDPOINT = "tcp://coordinator:5558"
    HANDSHAKE_ENDPOINT = "tcp://coordinator:5560"

    PRIVATE_KEY_PATH = os.path.join("keys", f"worker{WORKER_ID}_private.pem")
    COORDINATOR_PUBLIC_KEY_PATH = os.path.join("keys", "coordinator_public.pem")
    # If set to an integer N (>0), the worker will intentionally misbehave
    # (produce an incorrect hash) for the first N tasks it processes. This
    # is useful for testing coordinator verification and logging.
    MISBEHAVE_COUNT = int(os.environ.get("WORKER_MISBEHAVE", 0))
