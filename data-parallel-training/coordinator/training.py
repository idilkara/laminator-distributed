import numpy as np
import torch

# ---------- Task Distribution ----------
def batches_for_workers(X, y, num_workers, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    chunks = np.array_split(idx, num_workers)
    for wid, chunk_idx in enumerate(chunks):
        yield wid, X[chunk_idx], y[chunk_idx]


def compute_grad_and_loss(task):
    import torch.nn as nn
    from models import ModelHandler

    X = torch.tensor(task["X"], dtype=torch.float32)
    y = torch.tensor(task["y"], dtype=torch.long)
    state_dict = task["weights"]
    model_string = task["model_string"]

    model = ModelHandler.parse_model_string(model_string)
    assert model != -1

    # Load weights (convert numpy to tensor if needed)
    coerced = {k: torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    model.load_state_dict(coerced)
    model.train()

    criterion = nn.CrossEntropyLoss()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()

    grads = {name: p.grad.clone().numpy() for name, p in model.named_parameters()}

    return grads, float(loss.item()), X.shape[0]
