from grid_env_multi import RiskyGrid11, DelayedGrid11, DistractGrid11
from dqn_agent import DQNAgent
from emotion_module_ex5 import EmotionModuleEx5

import torch
import numpy as np
import pickle

TASKS = {
    "risky": RiskyGrid11,
    "delayed": DelayedGrid11,
    "distract": DistractGrid11,
}

def run_task(task_name, use_emotion=True, episodes=300, lambda_emo=2.0):
    env = TASKS[task_name]()
    agent = DQNAgent(state_dim=6, action_dim=4)
    emo = EmotionModuleEx5() if use_emotion else None

    eps = 1.0
    returns_list = []
    emo_list = []
    fail_list = []

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

            if use_emotion:
                R_emo = emo(torch.tensor([s], dtype=torch.float32)).item()
                R = R_ext + lambda_emo * R_emo
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
        emo_list.append(emo_sum)
        fail_list.append(failed)

        tag = "emo" if use_emotion else "base"
        print(f"{task_name}-{tag} | EP {ep+1} | Return={total:.2f} | Emo={emo_sum:.2f} | fail={failed} | eps={eps:.3f}")

    tag = "emo" if use_emotion else "base"
    fname = f"ex5_{task_name}_{tag}.pkl"
    pickle.dump((returns_list, emo_list, fail_list), open(fname, "wb"))
    print("SAVED:", fname)


if __name__ == "__main__":
    # run all tasks for baseline and emotional
    for name in TASKS.keys():
        run_task(name, use_emotion=False)
        run_task(name, use_emotion=True)