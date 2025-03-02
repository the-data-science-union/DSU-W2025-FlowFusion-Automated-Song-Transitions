import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=2048, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class AddNorm(nn.Module):
    def __init__(self, size, dropout_rate=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer))

