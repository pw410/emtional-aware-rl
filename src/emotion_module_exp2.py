import torch
import torch.nn as nn

def ramanujan(x):
    inner1 = torch.sqrt(torch.clamp(1 + 0.5 * x, min=1e-6))
    inner2 = torch.sqrt(torch.clamp(1 + 0.5 * inner1, min=1e-6))
    out = torch.sqrt(torch.clamp(1 + 0.5 * inner2, min=1e-6))
    return out

# ------------------------------------------------------------
# 1) Awareness OFF → I(t) = 1
# ------------------------------------------------------------
class Emotion_NoAwareness(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_enc = nn.Linear(2, 32)
        self.env_enc = nn.Linear(4, 32)
        self.ln = nn.LayerNorm(32)

        self.M = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(0.01 * torch.randn(32))

    def forward(self, state):
        s_self = state[:, :2]
        s_env  = state[:, 2:]

        z1 = self.ln(ramanujan(self.self_enc(s_self)))
        z2 = self.ln(ramanujan(self.env_enc(s_env)))

        h_raw = z1 + self.M * z2

        I = torch.ones_like(h_raw)          # <<< fixed awareness (OFF)
        h = I * h_raw

        E = torch.sum(h * self.w, dim=1)
        return torch.tanh(E)


# ------------------------------------------------------------
# 2) Awareness RANDOM → I(t) = random noise
# ------------------------------------------------------------
class Emotion_RandomAwareness(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_enc = nn.Linear(2, 32)
        self.env_enc = nn.Linear(4, 32)
        self.ln = nn.LayerNorm(32)

        self.M = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(0.01 * torch.randn(32))

    def forward(self, state):
        s_self = state[:, :2]
        s_env  = state[:, 2:]

        z1 = self.ln(ramanujan(self.self_enc(s_self)))
        z2 = self.ln(ramanujan(self.env_enc(s_env)))

        h_raw = z1 + self.M * z2

        I = torch.rand_like(h_raw)          # <<< random awareness (noise)
        h = I * h_raw

        E = torch.sum(h * self.w, dim=1)
        return torch.tanh(E)


# ------------------------------------------------------------
# 3) Awareness ON (learned) → SAME as previous Experiment-1
# ------------------------------------------------------------
class Emotion_LearnedAwareness(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_enc = nn.Linear(2, 32)
        self.env_enc = nn.Linear(4, 32)
        self.ln = nn.LayerNorm(32)

        self.M = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(0.01 * torch.randn(32))

        self.awareness = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )

    def forward(self, state):
        s_self = state[:, :2]
        s_env  = state[:, 2:]

        z1 = self.ln(ramanujan(self.self_enc(s_self)))
        z2 = self.ln(ramanujan(self.env_enc(s_env)))

        h_raw = z1 + self.M * z2

        I = self.awareness(torch.cat([z1, z2], dim=1))
        I = 0.5 + 0.5 * I
        h = I * h_raw

        E = torch.sum(h * self.w, dim=1)
        return torch.tanh(E)