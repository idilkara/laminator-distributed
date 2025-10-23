import argparse
import time
from dataclasses import dataclass
import torch, numpy as np, random, os
import pandas as pd
import zmq
import torch.nn as nn
from sklearn.model_selection import train_test_split

from models import ModelHandler

TASK_ENDPOINT = "tcp://*:5557"
RESULT_ENDPOINT = "tcp://*:5558"


@dataclass
class Config:
    num_workers: int
    epochs: int = 15
    lr: float = 0.1
    seed: int = 42


# ---------- Data Preprocessing ----------
def process_census(path="./data/adult.data"):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'martial_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target'
    ]

    df = pd.read_csv(
        path, names=column_names, na_values="?",
        sep=r'\s*,\s*', engine='python'
    ).loc[lambda d: d['race'].isin(['White', 'Black'])]

    # Binary sensitive attrs
    df['race'] = (df['race'] == 'White').astype(int)
    df['sex'] = (df['sex'] == 'Male').astype(int)

    y = (df['target'] == '>50K').astype(int)
    X = df.drop(columns=['target', 'race', 'sex', 'fnlwgt']).fillna('Unknown')
    X = pd.get_dummies(X, drop_first=True)

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

# ---------- Simple Neural Net ----------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# ---------- Task Distribution ----------
def batches_for_workers(X, y, num_workers, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    chunks = np.array_split(idx, num_workers)
    for wid, chunk_idx in enumerate(chunks):
        yield wid, X[chunk_idx], y[chunk_idx]


def send_task(sender, worker_id, Xb, yb, model_string, model, lr, epoch):
    task = {
        "worker_id": worker_id,
        "X": Xb.astype(np.float32).tolist(),
        "y": yb.astype(np.float32).tolist(),
        "architecture": model_string,
        "weights": model.state_dict(),   # <-- full NN state 
        "lr": float(lr),
        "epoch": int(epoch),
    }
    sender.send_pyobj(task)


def recv_results(collector, expected: int):
    grads_list, losses, counts = [], [], []
    for _ in range(expected):
        msg = collector.recv_pyobj()
        grads_list.append(msg["grads"])   # <-- dictionary of grads
        losses.append(msg["loss"])
        counts.append(msg["n"])

    total = np.sum(counts)

    # average gradients for each parameter name
    avg_grads = {}
    for name in grads_list[0].keys():
        avg_grads[name] = sum(
            torch.tensor(g[name]) * (c / total)
            for g, c in zip(grads_list, counts)
        )

    avg_loss = float(np.sum([l * (c / total) for l, c in zip(losses, counts)]))
    return avg_grads, avg_loss


# ---------- Training Loop ----------
def train(cfg: Config):
    ctx = zmq.Context.instance()
    task_out = ctx.socket(zmq.PUSH); task_out.bind(TASK_ENDPOINT)
    results_in = ctx.socket(zmq.PULL); results_in.bind(RESULT_ENDPOINT)

    # Data
    X_train, y_train, X_test, y_test = process_census()
    rng = np.random.default_rng(cfg.seed)

    # Initialize NN model
  

    # Initialize model
    # model = LinearNet([128, 256, 128])   # you can tweak sizes 
    model_string = '''
    {
        "model_type": "CustomizableLinearNet",
        "params": {
            "hidden_layer_sizes": [128, 256, 128],
            "input_dim": 93,
            "output_dim": 2,
            "activation": "tanh",
            "flatten": true
        }
    }
    '''

    # ASSUMPTION: worker has models.py file. This can be easily changed later but JSON strings will get ugly
    model = ModelHandler.parse_model_string(model_string) # To ensure that model is EXACTLY the same as worker side
    assert model != -1

    print(f"Coordinator started with {cfg.num_workers} workers. Data: X={X_train.shape}, y={y_train.shape}")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        for wid, Xb, yb in batches_for_workers(X_train, y_train, cfg.num_workers, rng):
            send_task(task_out, wid, Xb, yb, model_string, model, cfg.lr, epoch)

        avg_grads, avg_loss = recv_results(results_in, cfg.num_workers)

        # Apply gradients to model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param -= cfg.lr * avg_grads[name].to(param.device)

        dt = time.time() - t0

            # Evaluate on test set
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_test_t)              # shape: [N, 2]
            y_pred = torch.argmax(logits, dim=1).numpy()  # pick class with highest logit

        acc = np.mean(y_pred == y_test)
        print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] loss={avg_loss:.4f} | time={dt:.2f}s | Acc={acc:.4f}", flush=True)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_t)              # shape: [N, 2]
        y_pred = torch.argmax(logits, dim=1).numpy()  # pick class with highest logit

    acc = np.mean(y_pred == y_test)
    print(f"Final test Accuracy: {acc:.4f}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force determinism (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    cfg = Config(num_workers=args.num_workers, epochs=args.epochs, lr=args.lr, seed=args.seed)
    train(cfg)


if __name__ == "__main__":
    main()
