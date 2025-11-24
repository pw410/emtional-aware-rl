from grid_env_social import GridWorldSocial
from dqn_agent import DQNAgent
from emotion_module_exp4 import Emotion_MirrorOff, Emotion_MirrorOn

import torch
import pickle
import numpy as np

env = GridWorldSocial()

# -----------------------
# CHOOSE CONDITION HERE:
# -----------------------
emo = Emotion_MirrorOff()
cond_name = "mirror_off"

#emo = Emotion_MirrorOn()
#cond_name = "mirror_on"
# -----------------------

agent = DQNAgent(state_dim=9, action_dim=4)

lambda_emo = 2.0
episodes = 300
eps = 1.0

returns_list = []
emo_list = []
fail_list = []   # our agent lava = 1, else 0

for ep in range(episodes):
    s = env.reset()
    total = 0.0
    emo_sum = 0.0
    failed = 0

    for t in range(200):
        a = agent.act(s, eps)

        ns, R_ext, done = env.step(a)

        if R_ext <= -10.0:
            failed = 1

        state_t = torch.tensor([s], dtype=torch.float32)
        R_emo = emo(state_t).item()

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
    fail_list.append(failed)

    print(f"{cond_name} | EP {ep+1} | Return={total:.2f} | Emo={emo_sum:.2f} | fail={failed} | eps={eps:.3f}")

data_file = f"ex4_{cond_name}.pkl"
pickle.dump((returns_list, emo_list, fail_list), open(data_file, "wb"))
print("SAVED:", data_file)