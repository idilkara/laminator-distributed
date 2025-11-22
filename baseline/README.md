# Baseline

Quick start
1. From this folder run (builds images and starts services in background):

```zsh
docker compose up -d --build
```

2. Check logs (example: coordinator):

```zsh
docker compose logs -f coordinator
```

3. Stop and remove containers and network created by compose

```zsh
docker compose down
```

This prints a timing summary to the terminal after the stack is stopped.

Automated run logging
---------------------

Use `collect_run.py` to automate repeated baseline runs and capture the
coordinator logs in a single file:

```bash
python collect_run.py              # append one run to run_log.txt
python collect_run.py --runs 5     # append five runs back-to-back
```

Pass `--output custom.txt` to store logs in a different file or `--project-dir`
if you moved the compose stack elsewhere.

Configuration

- MODEL: modify the `model_string` variable in `coordinator/coordinator.py` to change the example model used by the demo.

To change the number of workers

1. Add one or more worker service entries to `docker-compose.yaml`. Example (adds a second worker):

```yaml
  worker2:
    build: .
    container_name: baseline-worker2
    command: python ./worker/worker.py
    environment:
      WORKER_ID: "1"
    networks:
      - training-net
    depends_on:
      - coordinator
```

2. Update the coordinator service arguments to match the total number of workers. Example coordinator snippet:

```yaml
  coordinator:
    build: .
    container_name: baseline-coordinator
    command: >
      python ./coordinator/coordinator.py
      --num-workers 2
      --epochs 10
      --lr 0.1
```

Notes
- Make sure `--num-workers` equals the total number of worker services you add in the compose file.
- Instead of hard-coding `container_name` values, consider using a project name or an `.env` file to avoid name collisions when running multiple stacks.
