from grid_env import GridWorld11x11
from dqn_agent import DQNAgent

from emotion_module_exp3 import (
    Emotion_WrongWeights,
    Emotion_NeutralWeights,
    Emotion_LearnedWeights
)

import torch
import numpy as np
import pickle

env = GridWorld11x11()

# -----------------------
# SELECT ONE:
# -----------------------
# emo = Emotion_WrongWeights()
# emo = Emotion_NeutralWeights()
emo = Emotion_LearnedWeights()   # default testing
# -----------------------

agent = DQNAgent(state_dim=6, action_dim=4)

lambda_emo = 2.0
episodes = 300
eps = 1.0

returns_list = []
emo_list = []
avoid_list = []   # how many times agent avoids 5,5

for ep in range(episodes):
    s = env.reset()
    total = 0
    emo_sum = 0
    avoided = 0

    for t in range(200):
        a = agent.act(s, eps)
        ns, R_ext, done = env.step(a)

        # fear avoidance: agent never touches lava
        if tuple(env.agent) == env.lava:
            avoid = 0
        else:
            avoid = 1
        avoided += avoid

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
    avoid_list.append(avoided)

    print(f"EP {ep+1} | Return={total:.2f} | Emo={emo_sum:.2f} | Avoid={avoided}")

pickle.dump((returns_list, emo_list, avoid_list), open("ex3_data.pkl", "wb"))