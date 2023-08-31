import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import GPT2ForQuestionAnswering
#def init_param(model):
#    "Initialize the parameters of resnet."
#    if isinstance(model, (nn.BatchNorm2d, nn.InstanceNorm2d)):
#        model.weight.data.fill_(1)
#       model.bias.data.zero_()
#   elif isinstance(model, nn.Linear):
#       model.bias.data.zero_()
#   return model
class GPTModelWrapper(nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
       # self.model.apply(init_param)

    def forward(self, feature):
        "Forward function."
        return self.model(feature)