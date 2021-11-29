#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from d3m.metadata import base as metadata_base

CYTHON_DEP = {
    "type": metadata_base.PrimitiveInstallationType.PIP,
    "package": "Cython",
    "version": "0.29.24",
}


def maybe_subset(X, y, n):
    if (n > 0) and (n < X.shape[0]):
        sel = np.sort(np.random.choice(X.shape[0], n, replace=False))
        return X[sel], y[sel]
    else:
        return X, y


def parmap(fn, x, n_jobs=1, backend="loky", verbose=1, **kwargs):
    if len(list(x)) < n_jobs:  # TODO: I'm surprised this is necessary
        n_jobs = len(list(x))

    jobs = [delayed(fn)(xx, **kwargs) for xx in x]
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(jobs)


class Img2Vec:
    def __init__(
        self,
        model_path,
        model="resnet-18",
        layer="default",
        layer_output_size=512,
        device="cuda",
    ):
        """Img2Vec
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        :param device: String that lets us decide between using cpu and gpu
        """
        self.device = device
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(
            model, layer, model_path
        )

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = (
            self.normalize(self.to_tensor(self.scaler(img)))
            .unsqueeze(0)
            .to(self.device)
        )

        if self.model_name == "alexnet":
            my_embedding = torch.zeros(1, self.layer_output_size)
        else:
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            if self.model_name == "alexnet":
                return my_embedding.numpy()[0, :]
            else:
                return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer, model_path):
        """Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == "resnet-18":
            model = models.resnet18()
            model.load_state_dict(torch.load(model_path))
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer
        else:
            raise KeyError("Model %s was not found" % model_name)
        """
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer
        """
