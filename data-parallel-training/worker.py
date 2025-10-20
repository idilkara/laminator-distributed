import torch
import torch.nn as nn
import torch.nn.functional as F
import zmq
import time, random
import numpy as np

TASK_ENDPOINT = "tcp://coordinator:5557"
RESULT_ENDPOINT = "tcp://coordinator:5558"

# A simple NN 
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

from models import LinearNet


def parse_model_string(model_string):
    blocks = model_string.split(';')
    for block in blocks:
        split = block.split(':')
        layer_name = split[0]
        # translate layer name into nn.layer
        layer_function = None #nn.Linear
        if(layer_name=="LinearNet"):
            layer_function=nn.Linear
        layer_sizes = np.fromstring(split[1].strip('[]'), dtype=int, sep=',')
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layers.append(layer_function(in_features, out_features))

    return nn.Sequential(*layers)


def compute_grad_and_loss(task):
    X = torch.tensor(task["X"], dtype=torch.float32)
    y = torch.tensor(task["y"], dtype=torch.long)  # classification labels
    state_dict = task["weights"]
    model_string = task["architecture"]

    # hash architecture
    # hash initial weights
    # hash training data

    # model = LinearNet([128, 256, 128]) # Have this be sent over as well. AS binary object or text based (like JSON or XML) and built on worker end
    model=parse_model_string(model_string)
    model.load_state_dict(state_dict)
    model.train()

    criterion = nn.CrossEntropyLoss()
    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()

    grads = {name: p.grad.clone().numpy() for name, p in model.named_parameters()}

    # hash grads

    return grads, float(loss.item()), X.shape[0] # also return hashes

def main():
    ctx = zmq.Context()
    receiver = ctx.socket(zmq.PULL); receiver.connect(TASK_ENDPOINT)
    sender = ctx.socket(zmq.PUSH); sender.connect(RESULT_ENDPOINT)

    time.sleep(1)

    while True:
        task = receiver.recv_pyobj()
        print("Received task from the coordinator",flush=True)
        grads, loss, n = compute_grad_and_loss(task)
        time.sleep(random.uniform(0.05, 0.2))
        sender.send_pyobj({
            "worker_id": task["worker_id"],
            "grads": grads,
            "loss": loss,
            "n": n,
        })
        # After the computation is complete, a hash of the output is incorporated into the user data field of an SGX report, which is then
        # used to obtain a DCAP quote [ 28 ] that can be verified by remote parties. This output takes the form of a JSON string representing
        # a fragment of the model card metadata4, where a model is named with the hash of its file.

if __name__ == "__main__":
    main()
