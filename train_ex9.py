from grid_env_social import GridWorldSocial
from dqn_agent import DQNAgent
from emotion_module_ex9 import (
    Emotion_MirrorOff_Aware,
    Emotion_MirrorOn_NoAware,
    Emotion_MirrorOn_Aware,
)

import torch
import numpy as np
import pickle

EPISODES = 300
MAX_STEPS = 200
LAMBDA_EMO = 2.0

CONDITIONS = [
    ("mirror_off_aware",  Emotion_MirrorOff_Aware),
    ("mirror_on_noaware", Emotion_MirrorOn_NoAware),
    ("mirror_on_aware",   Emotion_MirrorOn_Aware),
]


def run_condition(name, EmoCls):
    env = GridWorldSocial()
    agent = DQNAgent(state_dim=9, action_dim=4)
    emo = EmoCls()

    eps = 1.0
    returns = []
    emos = []
    fails = []
    contagion = []   # 1 if other hurt & self fail in same episode

    for ep in range(EPISODES):
        s = env.reset()
        total = 0.0
        emo_sum = 0.0
        self_fail = 0
        other_hurt = 0

        for t in range(MAX_STEPS):
            a = agent.act(s, eps)
            ns, R_ext, done = env.step(a)

            # self failure?
            if R_ext <= -10.0:
                self_fail = 1

            # other agent hurt?
            if env.other_event < 0:
                other_hurt = 1

            # emotional reward
            state_t = torch.tensor([s], dtype=torch.float32)
            R_emo = emo(state_t).item()
            R = R_ext + LAMBDA_EMO * R_emo

            agent.store(s, a, R, ns, float(done))
            agent.update()

            s = ns
            total += R
            emo_sum += R_emo

            if done:
                break

        eps = max(0.05, eps * 0.97)

        returns.append(total)
        emos.append(emo_sum)
        fails.append(self_fail)
        contagion.append(1 if (other_hurt and self_fail) else 0)

        print(f"{name} | EP {ep+1} | Ret={total:.2f} | Emo={emo_sum:.2f} | fail={self_fail} | otherHurt={other_hurt} | contagion={contagion[-1]} | eps={eps:.3f}")

    data = dict(
        returns=np.array(returns, dtype=np.float32),
        emos=np.array(emos, dtype=np.float32),
        fails=np.array(fails, dtype=np.float32),
        contagion=np.array(contagion, dtype=np.float32),
    )
    fname = f"ex9_{name}.pkl"
    pickle.dump(data, open(fname, "wb"))
    print("SAVED:", fname)


if __name__ == "__main__":
    for name, cls in CONDITIONS:
        run_condition(name, cls)