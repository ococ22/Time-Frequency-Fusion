import os
from model import SimpleTM
import torch

# Add this at the beginning of your training script
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True

import numpy as np

class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model_dict = {
            'SeqScale': SeqScale  # Map model name to SeqScale class
        }
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            if self.args.use_multi_gpu:
                print('Use multi GPU', self.args.device_ids)
                device = torch.device('cuda:{}'.format(self.args.device_ids[0]))
            print('Use GPU: cuda')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        return self.model_dict[self.args.model](self.args).float()
