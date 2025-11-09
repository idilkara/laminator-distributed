# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import json 

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class LinearNet(nn.Module):

    def __init__(self,hidden_layer_sizes = [128, 256, 128]):
        super().__init__()

        layers = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if i == 0:
                layers += [nn.Flatten()]
                layers += [nn.Linear(93, hidden_size)]
                layers += [nn.Tanh()]
            else:
                layers += [nn.Linear(hidden_layer_sizes[i - 1], hidden_size)]
                layers += [nn.Tanh()]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layer_sizes[-1], 2)

    def forward(self, x: torch.Tensor):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


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