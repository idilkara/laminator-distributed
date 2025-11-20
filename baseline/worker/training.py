from models import ModelHandler

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json

## for the oordinator Process


# A simple NN 

class ModelHandler:
    #-------Construct model from JSON representation--------------
    @staticmethod
    def parse_model_string(model_string):
        config = json.loads(model_string)
        if(config["model_type"] == "CustomizableLinearNet"):
            return CustomizableLinearNet(**config["params"])
        else:
            print("Unsupported model type: ", config["model_type"])
            return -1

class CustomizableLinearNet(nn.Module):
    def __init__(self, hidden_layer_sizes=[128, 256, 128], input_dim=93, output_dim=2, activation="tanh", flatten=True):
        super().__init__()
        layers = []
        if flatten:
            layers.append(nn.Flatten())

        activ_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid
        }[activation]

        for i, hidden_size in enumerate(hidden_layer_sizes):
            in_features = input_dim if i == 0 else hidden_layer_sizes[i - 1]
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activ_fn())

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layer_sizes[-1], output_dim)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)



def compute_grad_and_loss(task):
    X = torch.tensor(task["X"], dtype=torch.float32)
    y = torch.tensor(task["y"], dtype=torch.long)  # classification labels
    state_dict = task["weights"]
    model_string = task["architecture"]

    # hash architecture
    # hash initial weights
    # hash training data

    model=ModelHandler.parse_model_string(model_string)
    assert model != -1

    # Ensure state_dict values are torch.Tensors. Some transports may
    # deserialize tensors as numpy arrays; convert them back to tensors here.
    coerced = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            coerced[k] = v
        else:
            try:
                # numpy arrays or lists -> tensor
                coerced[k] = torch.as_tensor(v)
            except Exception:
                # Fallback: use original value and let load_state_dict raise
                coerced[k] = v

    model.load_state_dict(coerced)
    model.train()

    criterion = nn.CrossEntropyLoss()
    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()

    grads = {name: p.grad.clone().numpy() for name, p in model.named_parameters()}

    # perform a single SGD step using provided learning rate (if present)
    lr = float(task.get("lr", 0.01))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            p.add_( - lr * p.grad )

    # collect updated weights as numpy arrays
    updated_state = {name: p.detach().cpu().numpy() for name, p in model.named_parameters()}

    return grads, float(loss.item()), X.shape[0], updated_state # also return updated model
