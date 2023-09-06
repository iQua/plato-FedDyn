import copy
import logging
import multiprocessing as mp
import os
import pickle
import re
import time

import torch
from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import LogProgressCallback
from plato.config import Config 
from plato.models import registry as models_registry
from plato.trainers import basic,base, loss_criterion, lr_schedulers, optimizers, tracking, huggingface
from transformers import AutoTokenizer,GPT2ForQuestionAnswering
class Trainer(huggingface.Trainer):

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        self.optimizer.zero_grad()
        outputs = self.model(examples.int())

        loss = outputs.loss
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss