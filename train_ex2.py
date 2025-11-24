import pickle
import numpy as np
from grid_env import GridWorld11x11
from dqn_agent import DQNAgent
from emotion_module import EmotionModule
import torch


# -----------------------------------------------------
# Run function for 1 agent setup
# -----------------------------------------------------
def run_agent(awareness_on=False, emotion_on=False):
    print(f"\nRunning Awareness={awareness_on}   Emotion={emotion_on} ...")

    env = GridWorld11x11()
    state_dim = env.state_dim
    action_dim = env.action_space

    # Agent
    agent = DQNAgent(state_dim, action_dim, awareness_on=awareness_on)

    # Emotion module
    emo = EmotionModule()

    episodes = 300
    epsilon = 1.0

    returns = []

    for ep in range(episodes):
        s = env.reset()
        total = 0

        for step in range(200):

            a = agent.act(s, epsilon)
            ns, r_env, done = env.step(a)

            r = r_env

            # ----- emotional reward -----
            if emotion_on:
                s_t = torch.tensor(s, dtype=torch.float32)
                emo_val = emo(s_t.unsqueeze(0)).item()
                r += emo_val

            # ----- awareness system -----
            if agent.awareness_on:
                # awareness bonus = +0.1 * ||state||
                aware = 0.1 * np.linalg.norm(s)
                r += aware

            agent.store(s, a, r, ns, done)
            agent.update()

            total += r
            s = ns

            if done:
                break

        epsilon = max(0.05, epsilon * 0.97)
        returns.append(total)

        print(f"EP {ep+1} | Return = {total:.2f} | eps={epsilon:.3f}")

    return np.array(returns)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":

    # 1) BASELINE (no awareness, no emotion)
    base = run_agent(False, False)
    pickle.dump(base, open("ex2_base.pkl", "wb"))

    # 2) EMOTION ONLY
    emo = run_agent(False, True)
    pickle.dump(emo, open("ex2_emo.pkl", "wb"))

    # 3) AWARENESS ONLY
    aware = run_agent(True, False)
    pickle.dump(aware, open("ex2_aware.pkl", "wb"))

    # 4) EMOTION + AWARENESS
    both = run_agent(True, True)
    pickle.dump(both, open("ex2_emo_aware.pkl", "wb"))

    print("\n=== Experiment 2 DONE. Data Saved. ===")