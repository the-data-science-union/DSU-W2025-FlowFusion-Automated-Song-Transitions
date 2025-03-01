import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaPlus(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super(MambaPlus, self).__init__()
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        self.forget_gate = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        new_features = self.mamba(x)
        forget_gate = torch.sigmoid(self.forget_gate(x))
        return forget_gate * x + (1 - forget_gate) * new_features