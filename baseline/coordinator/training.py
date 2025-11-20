import numpy as np
import torch

# ---------- Task Distribution ----------
def batches_for_workers(X, y, num_workers, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    chunks = np.array_split(idx, num_workers)
    for wid, chunk_idx in enumerate(chunks):
        yield wid, X[chunk_idx], y[chunk_idx]


def _avg_grads_and_loss(grads_list, counts_list, losses_list):
    """
    grads_list: list[dict[param_name -> list[float]]]  (JSON from workers)
    counts_list: list[int]  (per-worker number of samples)
    losses_list: list[float]

    Returns:
      avg_grads: dict[param_name -> torch.Tensor]
      avg_loss: float (weighted by counts)
    """
    if not grads_list:
        return {}, 0.0

    # Union of param names across workers
    param_names = set()
    for gd in grads_list:
        param_names.update(gd.keys())

    total_n = float(sum(int(n) for n in counts_list))
    if total_n <= 0:
        total_n = 1.0

    # Weighted average of grads by sample count
    avg_grads = {}
    for name in sorted(param_names):
        acc = None
        for gd, n in zip(grads_list, counts_list):
            if name not in gd:
                continue
            g = torch.tensor(gd[name], dtype=torch.float32)
            weight = float(n) / total_n
            acc = g * weight if acc is None else acc + g * weight
        if acc is None:
            # If no worker provided this param (shouldn't happen), set zeros
            acc = torch.tensor(0.0)
        avg_grads[name] = acc

    # Weighted average loss
    wloss = 0.0
    for n, l in zip(counts_list, losses_list):
        wloss += float(l) * (float(n) / total_n)
    return avg_grads, float(wloss)
