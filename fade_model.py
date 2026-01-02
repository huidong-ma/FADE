import torch
import torch.nn as nn
import torch.nn.functional as F

class FADE(nn.Module):
    def __init__(self, batch_size=512, timesteps=16, vocab_dim=32, vocab_size=256):
        super(FADE, self).__init__()
        branch = 2
        self.batch_size = batch_size 

        self.input_map = nn.Embedding(vocab_size, vocab_dim)
        nn.init.normal_(self.input_map.weight, 0, 0.01)
        
        self.output_logit_map = nn.Linear(timesteps * vocab_dim, vocab_size)
        nn.init.normal_(self.output_logit_map.weight, 0, 0.01)
        nn.init.zeros_(self.output_logit_map.bias)

        self.local_feature = LocalCNNStream(timesteps, vocab_dim, batch_size)
        self.global_feature = GlobalMLPStream(timesteps, vocab_dim, ffn_dim=4096, batch_size=batch_size, last_dim=64)
        self.coarse = CoarseRefinement(branch, timesteps, vocab_dim, batch_size=batch_size)
        self.fine = FineRefinement(timesteps, vocab_dim, ffn_dim=8192, batch_size=batch_size)
        self.gate = nn.Sequential(
            nn.Linear(timesteps * vocab_dim, timesteps * vocab_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(timesteps * vocab_dim, eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        x_emb = self.input_map(x)
        x = self.norm(x_emb.reshape(self.batch_size, 1, -1))
        skip = x
        x_global = self.global_feature(x)
        x_local = self.local_feature(x_emb)
        gate = self.gate(skip)
        x = gate * x_global + (1 - gate) * x_local
        x = self.coarse(x)
        x = self.fine(x)
        x = self.output_logit_map(x)
        return x

class LocalCNNStream(nn.Module):
    def __init__(self, timesteps, vocab_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.vocab_dim = vocab_dim
        self.norm = nn.LayerNorm(vocab_dim, eps=1e-05)
        self.conv = nn.Conv1d(
            in_channels=vocab_dim, 
            out_channels=vocab_dim, 
            kernel_size=3, 
            padding=1,     
            groups=vocab_dim,      
            bias=True
        )
    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)
        return x.reshape(self.batch_size, 1, -1)

class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias=bias)
    def forward(self, x):
        x = self.linear(x)
        x_main, x_gate = x.chunk(2, dim=-1)
        return F.gelu(x_main) * x_gate

class GlobalMLPStream(nn.Module):
    def __init__(self, timesteps, vocab_dim, ffn_dim, batch_size, last_dim=64):
        super().__init__()
        self.batch_size = batch_size
        self.last_dim = last_dim
        self.vocab_dim = vocab_dim
        total_dim = timesteps * vocab_dim
        self.full = GeGLU(total_dim, ffn_dim, bias=True)
        self.last = GeGLU(vocab_dim, last_dim, bias=True)
        self.out = nn.Linear(ffn_dim, total_dim, bias=True)
        self.skip_scale = nn.Parameter(torch.tensor(1.0))
        self.cache = None
    
    def forward(self, x):
        skip = x
        if self.cache is None:
            full_out = self.full(x)
            self.cache = full_out.detach()
        else:
            last_out = self.last(x[:, :, -self.vocab_dim:])
            self.cache = torch.cat(
                (self.cache[:, :, self.last_dim:], last_out),
                dim=2
            ).detach()
        x = self.out(self.cache)
        x = F.gelu(x)
        x = x + skip * self.skip_scale
        return x

class CoarseRefinement(nn.Module):
    def __init__(self, branch, timesteps, vocab_dim, batch_size, hidden_dim=64, r=4):
        super().__init__()
        self.batch_size = batch_size      # batchsize
        self.branch = branch
        branch_dim = (timesteps * vocab_dim) // branch
        self.norm = nn.LayerNorm(branch_dim, eps=1e-05, elementwise_affine=True)
        self.down = nn.Linear(branch_dim, hidden_dim, bias=True)
        self.U = nn.Parameter(torch.normal(0, 0.01, (self.batch_size, hidden_dim, hidden_dim)), requires_grad=True)
        self.up = nn.Linear(hidden_dim, branch_dim, bias=True)
        self.bias = nn.Parameter(torch.normal(0, 0.01, (self.batch_size, self.branch, branch_dim)), requires_grad=True)
        self.skip_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim, branch_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(self.batch_size, self.branch, -1)
        x = self.norm(x)
        skip = x
        x = self.down(x)
        x = torch.bmm(x, self.U)
        x = self.up(x)
        x = x * self.gate(skip)
        x = x + self.bias
        x = x + skip * self.skip_scale
        x = x.reshape(self.batch_size, 1, -1)
        return x

class FineRefinement(nn.Module):
    def __init__(self, timesteps, vocab_dim, ffn_dim, batch_size):
        super().__init__()
        total_dim = timesteps * vocab_dim
        self.batch_size = batch_size
        self.glu = GeGLU(total_dim, ffn_dim)
        self.out = nn.Linear(ffn_dim, total_dim, bias=True)
        self.norm1 = nn.LayerNorm(total_dim, eps=1e-05, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(total_dim, eps=1e-05, elementwise_affine=True)
        self.skip_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.norm1(x)
        skip = x
        x = self.glu(x)
        x = self.out(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = x + skip * self.skip_scale
        return x
