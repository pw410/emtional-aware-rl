from grid_env import GridWorld11x11
from dqn_agent import DQNAgent
from emotion_module import EmotionModule
import torch
import numpy as np
import pickle

env = GridWorld11x11()

agent = DQNAgent(state_dim=6, action_dim=4)
emo = EmotionModule()

lambda_emo = 2.0
episodes = 300
eps = 1.0

returns_list = []
emo_list = []

for ep in range(episodes):
    s = env.reset()
    total = 0
    emo_sum = 0

    for t in range(200):
        a = agent.act(s, eps)
        ns, R_ext, done = env.step(a)

        R_emo = emo(torch.tensor([s], dtype=torch.float32)).item()
        R = R_ext + lambda_emo * R_emo

        agent.store(s, a, R, ns, float(done))
        agent.update()

        s = ns
        total += R
        emo_sum += R_emo

        if done:
            break

    eps = max(0.05, eps * 0.97)

    returns_list.append(total)
    emo_list.append(emo_sum)

    print(f"EP {ep+1} | Return={total:.2f} | Emo={emo_sum:.2f} | eps={eps:.3f}")
