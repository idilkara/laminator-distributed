from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import os
import binascii

class CoordinatorConfig:
    def __init__(self, num_workers, epochs, lr, seed):
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.task_endpoint = "tcp://*:5557"
        self.result_endpoint = "tcp://*:5558"
        self.registration_endpoint = "tcp://*:5560"

        self.private_key = X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def get_public_bytes(self):
        return self.public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw
        )


class WorkerConfig:
    def __init__(self, worker_id):
        self.worker_id = int(os.environ.get("WORKER_ID", 0))

        self.task_endpoint = "tcp://coordinator:5557"
        self.result_endpoint = "tcp://coordinator:5558"
        self.registration_endpoint = "tcp://coordinator:5560"

        self.private_key = X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def get_public_bytes(self):
        return self.public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw
        )

    def compute_shared_secret(self, coordinator_pub_bytes):
        coordinator_pub = X25519PublicKey.from_public_bytes(coordinator_pub_bytes)
        shared_secret = self.private_key.exchange(coordinator_pub)
        # Optionally derive an HMAC key from the shared secret:
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"federated-learning",
        ).derive(shared_secret)
        return derived_key
