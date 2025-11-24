from grid_env_multi import RiskyGrid11
from dqn_agent import DQNAgent
from emotion_module_ex5 import EmotionModuleEx5  # same as Exp5

import torch
import numpy as np
import pickle

NOISE_LEVELS = [0.0, 0.5, 1.0]   # 0 = clean, others = noisy
P_FLIP = 0.1                     # 10% chance reward sign flip
EPISODES = 300
MAX_STEPS = 200

def run_noise_level(noise_std, use_emotion=True, lambda_emo=2.0):
    tag = "emo" if use_emotion else "base"
    env = RiskyGrid11()
    agent = DQNAgent(state_dim=6, action_dim=4)
    emo = EmotionModuleEx5() if use_emotion else None

    eps = 1.0
    returns_list = []
    emo_list = []
    fail_list = []

    for ep in range(EPISODES):
        s = env.reset()
        total = 0.0
        emo_sum = 0.0
        failed = 0

        for t in range(MAX_STEPS):
            a = agent.act(s, eps)
            ns, R_ext, done = env.step(a)

            # mark failure on real reward
            if R_ext <= -10.0:
                failed = 1

            # ---- add noise & stress ----
            noisy = R_ext + np.random.randn() * noise_std
            if np.random.rand() < P_FLIP:
                noisy = -noisy    # stress spike / flip

            R_effective_ext = noisy

            # emotional version
            if use_emotion:
                R_emo = emo(torch.tensor([s], dtype=torch.float32)).item()
                R = R_effective_ext + lambda_emo * R_emo
                emo_sum += R_emo
            else:
                R = R_effective_ext

            agent.store(s, a, R, ns, float(done))
            agent.update()

            s = ns
            total += R

            if done:
                break

        eps = max(0.05, eps * 0.97)

        returns_list.append(total)
        emo_list.append(emo_sum)
        fail_list.append(failed)

        print(f"noise={noise_std:.2f}-{tag} | EP {ep+1} | Ret={total:.2f} | Emo={emo_sum:.2f} | fail={failed} | eps={eps:.3f}")

    fname = f"ex6_noise{noise_std}_{tag}.pkl"
    pickle.dump((returns_list, emo_list, fail_list), open(fname, "wb"))
    print("SAVED:", fname)


if __name__ == "__main__":
    for nl in NOISE_LEVELS:
        # baseline
        run_noise_level(nl, use_emotion=False)
        # emotional
        run_noise_level(nl, use_emotion=True)