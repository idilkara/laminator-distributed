Data-parallel training demo (coordinator + workers)
=================================================

This folder contains a small, self-contained demo of a coordinator + worker
system that performs data-parallel training. The coordinator sends signed
training tasks (model architecture, initial weights, batch data, config)
to workers over ZMQ. Workers compute gradients (and a one-step local update)
and return signed, hashed results. The coordinator verifies signatures and
hashes, aggregates gradients, updates the global model, and records a
verification report.

Quick start (Docker)
--------------------

1. Build and run the coordinator+workers with docker-compose (from this
     directory):

```bash
docker compose up --build
```

2. Compose sets default runtime arguments for the coordinator. To change
     training parameters edit `docker-compose.yaml` or run the coordinator
     locally (see below).

Important files
---------------

- `coordinator/` — coordinator process. Main file: `coordinator.py`.
- `worker/` — worker process. Main file: `worker.py`.
- `auth.py` — signing/verification helpers used by both coordinator and
    worker (in the service directories).
- `generate-keys.py` — helper to create RSA keypairs used by the demo.
- `data/` — small dataset used by the demo (mounted into the coordinator
    container).
- `keys/` — generated keys (coordinator and workers). Not checked into
    git by default.
- `docker-compose.yaml` — starts one coordinator and two workers by
    default (see `services:` entries).

Running locally (without Docker)
-------------------------------

1. Create keys (one-time):

```bash
python3 generate-keys.py
```

Change model & worker count
---------------------------

- Change the model used by the demo:

    Edit the `model_string` (or the model definition) in `coordinator/coordinator.py` to switch architectures or modify the example model. After changing the model code you can run the coordinator locally or rebuild the Docker image and restart the compose stack.

- Increase the number of workers (Docker):

    Add additional worker service entries in `docker-compose.yaml` and make sure the coordinator's `--num-workers` argument matches the total number of worker services. Example (adds a second worker):

    ```yaml
    worker2:
        build: .
        container_name: data-parallel-worker2
        command: python ./worker/worker.py
        environment:
            WORKER_ID: "1"
        networks:
            - training-net
        depends_on:
            - coordinator
    ```

    Then update the coordinator service args (or run the coordinator locally) so it knows how many workers to expect:

    ```yaml
    coordinator:
        build: .
        container_name: data-parallel-coordinator
        command: >
            python ./coordinator/coordinator.py
            --num-workers 2
            --epochs 10
            --lr 0.1
    ```
Notes
- Make sure `--num-workers` equals the number of worker services you defined in `docker-compose.yaml`.
- Instead of editing `docker-compose.yaml` directly


Verification / hashing behavior
------------------------------

- The coordinator computes stable SHA-256 hashes for:
    - the entire training dataset (H_dataset),
    - the model architecture JSON (H_arch),
    - the initial weights (H_weights_init), and
    - the training config (H_config).
- These global hashes are included at the top of `hash_report.txt`.
- For each per-worker task the coordinator computes per-task hashes as
    well (including a per-batch H_DTr). Workers compute the same hashes when
    generating results; the coordinator verifies both signatures and hashes
    and records mismatches in the report.

Testing failure modes (misbehaving workers)
------------------------------------------

For testing, workers support an opt-in misbehavior mode controlled by the
env var `WORKER_MISBEHAVE` (integer). Example in `docker-compose.yaml`:

```yaml
    worker1:
        environment:
            WORKER_ID: "0"
            WORKER_MISBEHAVE: "1"  # worker will corrupt first task's hash
        
```

When enabled the worker will intentionally corrupt one of the reported
hashes for the configured number of tasks. The coordinator will detect the
hash mismatch, ignore the result, and log the incident in the verification
report.

Where to look for logs and report
--------------------------------

- Coordinator stdout (or `docker logs coordinator`) shows:
    - handshake time, preprocessing time, per-epoch stats, and a timing summary.
    - the verification report contents (printed at the end)
- The report file appears at `data/hash_report.txt` on the host when using
    docker-compose.

Notes and next steps
--------------------

- This demo uses JSON serialization for portability. For large models or
    production workloads you would prefer binary formats and streaming.
- The current hashing and signature checks are demonstration-grade; in a
    production system you would also harden replay protection, nonce
    management, and key rotation.
- Use SGX instead of assuming. 



