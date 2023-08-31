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

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer(examples,labels, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        target_start_index = torch.tensor([14])
        target_end_index = torch.tensor([15])

        outputs = self.model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
        loss = outputs.loss
        self._loss_tracker.update(loss, labels.size(0))

        self.optimizer.step()

        return loss