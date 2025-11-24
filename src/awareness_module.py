import torch
import torch.nn as nn
import torch.nn.functional as F

class AwarenessGate(nn.Module):
    def __init__(self, state_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def compute(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        I = torch.sigmoid(self.out(x))
        return I.item()