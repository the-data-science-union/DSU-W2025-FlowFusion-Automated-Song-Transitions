from mamba_models.attention.encoder import Encoder
from mamba_models.mamba-plus import MambaPlus

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(BidirectionalMambaBlock, self).__init__()
        self.feature_dim = d_ff
        self.fmamba = MambaPlus(d_model, d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND)
        self.bmamba = MambaPlus(d_model, d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND)
        self.an1 = AddNorm(d_model, dropout)
        self.an2 = AddNorm(d_model, dropout)
        self.an3 = AddNorm(d_model, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x2 = self.fmamba(x)
        x2 = self.an1(x, x2)
        x3 = torch.flip(x)
        x3 = self.bmamba(x3)
        x3 = torch.flip(x3)
        x3 = self.an2(x, x3)
        x4 = x2 + x3
        x5 = self.feedforward(x4)
        x5 = self.an3(x4, x5)
        return x5