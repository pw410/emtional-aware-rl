import torch
import torch.nn as nn

def ramanujan(x):
    inner1 = torch.sqrt(torch.clamp(1 + 0.5 * x, min=1e-6))
    inner2 = torch.sqrt(torch.clamp(1 + 0.5 * inner1, min=1e-6))
    out = torch.sqrt(torch.clamp(1 + 0.5 * inner2, min=1e-6))
    return out

class EmotionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # encoders
        self.self_enc = nn.Linear(2, 32)   # (x,y)
        self.env_enc  = nn.Linear(4, 32)   # (lava, goal)

        self.ln = nn.LayerNorm(32)

        # mirror + subconscious weights
        self.M = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(0.01 * torch.randn(32))

        # awareness network
        self.awareness = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )

    def forward(self, state):
        # split input
        s_self = state[:, :2]
        s_env  = state[:, 2:]

        # encodings
        z1 = ramanujan(self.self_enc(s_self))
        z2 = ramanujan(self.env_enc(s_env))

        z1 = self.ln(z1)
        z2 = self.ln(z2)

        # raw emotion
        h_raw = z1 + self.M * z2

        # awareness gate
        I = self.awareness(torch.cat([z1, z2], dim=1))
        I = 0.5 + 0.5 * I    # warm-start awareness
        h = I * h_raw

        # emotional energy
        E = torch.sum(h * self.w, dim=1)

        # final emotional reward
        R_emo = torch.tanh(E)

        return R_emo