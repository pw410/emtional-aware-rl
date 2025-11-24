import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# -----------------------------
# Q-Network
# -----------------------------
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, awareness_on=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.awareness_on = awareness_on        # <<--- ADDED

        self.gamma = 0.99
        self.batch = 64
        self.memory = deque(maxlen=50000)

        self.q = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=1e-3)

    # -------------------------
    # Act (epsilon-greedy)
    # -------------------------
    def act(self, state, eps):
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)

        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(self.q(s).argmax().item())

    # -------------------------
    # Store experience
    # -------------------------
    def store(self, *args):
        self.memory.append(args)

    # -------------------------
    # Update Q-network
    # -------------------------
    def update(self):
        if len(self.memory) < self.batch:
            return

        batch = random.sample(self.memory, self.batch)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a).long()
        r = torch.tensor(r, dtype=torch.float32)
        ns = torch.tensor(ns, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        qvals = self.q(s)
        q_a = qvals[range(self.batch), a]

        with torch.no_grad():
            target = self.target(ns).max(1)[0]
            target_q = r + (1 - d) * 0.99 * target

        loss = nn.MSELoss()(q_a, target_q)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft update target network
        for p, tp in zip(self.q.parameters(), self.target.parameters()):
            tp.data.copy_(0.995 * tp.data + 0.005 * p.data)