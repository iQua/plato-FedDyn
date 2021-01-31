"""The LeNet-5 model for MindSpore.

Reference:

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to
document recognition." Proceedings of the IEEE, November 1998.
"""
import collections
import mindspore
import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import Normal

import models.base_mindspore as base_mindspore
from config import Config

class Model(base_mindspore.Model):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes.
    """
    def __init__(self, num_classes=10, cut_layer=None):
        super().__init__()
        self.cut_layer = cut_layer

        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Dense(84, num_classes, weight_init=Normal(0.02))

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['relu1'] = self.relu1
        self.layerdict['pool1'] = self.pool1
        self.layerdict['conv2'] = self.conv2
        self.layerdict['relu2'] = self.relu2
        self.layerdict['pool2'] = self.pool2
        self.layerdict['flatten'] = self.flatten
        self.layerdict['fc1'] = self.fc1
        self.layerdict['relu3'] = self.relu3
        self.layerdict['fc2'] = self.fc2
        self.layerdict['relu4'] = self.relu4
        self.layerdict['fc3'] = self.fc3
        self.layers.append('conv1')
        self.layers.append('relu1')
        self.layers.append('pool1')
        self.layers.append('conv2')
        self.layers.append('relu2')
        self.layers.append('pool2')
        self.layers.append('flatten')
        self.layers.append('fc1')
        self.layers.append('relu3')
        self.layers.append('fc2')
        self.layers.append('relu4')
        self.layers.append('fc3')

    def construct(self, x):
        """The forward pass."""
        if self.cut_layer is None:
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            x = self.fc3(x)
        else:
            """
            layer_index = self.layers.index(self.cut_layer)

            for i in range(layer_index + 1, len(self.layers)):
                print(self.layers[i])
                print(x.shape)
                x = self.layerdict[self.layers[i]](x)
            """
            print(x.shape)
            print(x)
            x = self.fc3(x)          
            print("Done!")   
        return x

    def forward_to(self, x, cut_layer):
        layer_index = self.layers.index(cut_layer)

        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)
        return x

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name == 'lenet5_mindspore'

    @staticmethod
    def get_model_from_name(model_name, **kwargs):
        """Obtaining an instance of this model provided that the name is valid."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        num_classes = 10
        if hasattr(Config().trainer, 'num_classes'):
            num_classes = Config().trainer.num_classes

        cut_layer = None

        if hasattr(Config().algorithm, 'cut_layer'):
            cut_layer = Config().algorithm.cut_layer

        if 'cut_layer' in kwargs:
            cut_layer = kwargs['cut_layer']

        return Model(num_classes=num_classes, cut_layer=cut_layer)
