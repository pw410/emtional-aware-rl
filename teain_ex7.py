from grid_env_multi import RiskyGrid11
from dqn_agent import DQNAgent
from emotion_module_ex5 import EmotionModuleEx5

import torch
import numpy as np
import random
import pickle

EPISODES = 300
MAX_STEPS = 200
N_SEEDS = 10
LAMBDA_EMO = 2.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_condition(use_emotion=True):
    """Run N_SEEDS times; return arrays [seeds, episodes]."""
    all_returns = []
    all_fail = []
    all_emo = []

    for seed in range(N_SEEDS):
        print(f"{'EMO' if use_emotion else 'BASE'} | Seed {seed}")
        set_seed(seed)

        env = RiskyGrid11()
        agent = DQNAgent(state_dim=6, action_dim=4)
        emo = EmotionModuleEx5() if use_emotion else None

        eps = 1.0
        returns_list = []
        fail_list = []
        emo_list = []

        for ep in range(EPISODES):
            s = env.reset()
            total = 0.0
            emo_sum = 0.0
            failed = 0

            for t in range(MAX_STEPS):
                a = agent.act(s, eps)
                ns, R_ext, done = env.step(a)

                if R_ext <= -10.0:
                    failed = 1

                if use_emotion:
                    R_emo = emo(torch.tensor([s], dtype=torch.float32)).item()
                    R = R_ext + LAMBDA_EMO * R_emo
                    emo_sum += R_emo
                else:
                    R = R_ext

                agent.store(s, a, R, ns, float(done))
                agent.update()

                s = ns
                total += R

                if done:
                    break

            eps = max(0.05, eps * 0.97)

            returns_list.append(total)
            fail_list.append(failed)
            emo_list.append(emo_sum)

            print(f"  EP {ep+1} | R={total:.2f} | Emo={emo_sum:.2f} | fail={failed} | eps={eps:.3f}")

        all_returns.append(returns_list)
        all_fail.append(fail_list)
        all_emo.append(emo_list)

    return (
        np.array(all_returns, dtype=np.float32),
        np.array(all_fail, dtype=np.float32),
        np.array(all_emo, dtype=np.float32),
    )


if __name__ == "__main__":
    # Baseline
    base_ret, base_fail, base_emo = run_condition(use_emotion=False)
    pickle.dump(
        (base_ret, base_fail, base_emo),
        open("ex7_base.pkl", "wb")
    )
    print("SAVED: ex7_base.pkl")

    # Emotional
    emo_ret, emo_fail, emo_emo = run_condition(use_emotion=True)
    pickle.dump(
        (emo_ret, emo_fail, emo_emo),
        open("ex7_emo.pkl", "wb")
    )
    print("SAVED: ex7_emo.pkl")