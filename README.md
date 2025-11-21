# Laminator Distributed (CS 858 Project)

This repository contains small demos of a distributed training coordinator + worker system implemented for the CS 858 project.

There are two example stacks in this repo:

- `baseline/` — one coordinator and one (or more) worker containers for data-parallel-training but without any attestation or assumptions. See `baseline/README.md`.
- `data-parallel-training/` — oone coordinator and one (or more) worker containers performing data-parallel training with signed, hashed tasks and results, assuming it was run inside TEEs. See `data-parallel-training/README.md`.

High-level flow

- The coordinator prepares training tasks that include the model architecture, initial weights, a batch of data, and training configuration.
- Tasks are dispatched to workers; workers perform local computation (compute gradients and a one-step local update across the configured number of epochs or steps) and return results to the coordinator.
- (for the second setup) Results are signed and hashed by workers; the coordinator verifies signatures and hashes, aggregates gradients, updates the global model, and records a verification report with the overall hashes that were input to the coordinator(`data/hash_report.txt`).
- Verifier can verify using the verify.py script in `data/verifier.py`. (also see the `data/README.md`) 


Using Docker

From the respective folder run:

```zsh
docker compose up --build
```

Stop and remove the stack with:

```zsh
docker compose down
```


Quick edits

- Change the example model: edit the model code or `model_string` in `coordinator/coordinator.py` (or the coordinator folder for the baseline demo), then rebuild the images and restart the stack.
- Increase the worker count: add worker service entries to the compose YAML and set the coordinator `--num-workers` argument to the total number of worker services (see both demo READMEs for copy-paste examples).

Security & notes

- This repository include pre-generated public and secret keys under `keys/` for convenience. 

- `baseline/README.md` - try the data-parallel training. 

- `data-parallel-training/README.md` — generate-keys helper, testing failure modes, and verification report behavior.

