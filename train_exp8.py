from grid_env_multi import RiskyGrid11
from dqn_agent import DQNAgent
from emotion_module_ex5 import EmotionModuleEx5

import torch
import numpy as np
import pickle

# ------------ CONFIG ----------------
EPISODES = 300
MAX_STEPS = 200

COND_EP = 100        # Phase 1: conditioning
EXT_EP = 100         # Phase 2: extinction
AWARE_EP = 100       # Phase 3: awareness healing

FEAR_TILE = (3, 3)   # "blue" tile
LAMBDA_EMO = 2.0
# ------------------------------------


env = RiskyGrid11()
agent = DQNAgent(state_dim=6, action_dim=4)
emo = EmotionModuleEx5()

eps = 1.0

returns = []
emos = []
fear_visits = []   # how many times fear tile visited per episode
phases = []        # 0=cond, 1=extinction, 2=awareness

for ep in range(EPISODES):
    s = env.reset()
    total = 0.0
    emo_sum = 0.0
    visit_count = 0

    # decide phase
    if ep < COND_EP:
        phase = 0    # conditioning
    elif ep < COND_EP + EXT_EP:
        phase = 1    # extinction
    else:
        phase = 2    # awareness healing

    for t in range(MAX_STEPS):
        a = agent.act(s, eps)
        ns, R_ext, done = env.step(a)

        # current agent position
        ax, ay = env.agent[0], env.agent[1]

        # fear tile visit
        if (ax, ay) == FEAR_TILE:
            visit_count += 1

            if phase == 0:
                # conditioning: strong shock when on fear tile
                R_ext = -10.0
                done = True

            elif phase == 2:
                # awareness phase: mild positive re-evaluation
                R_ext += 3.0  # exposure therapy style

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
    fear_visits.append(visit_count)
    phases.append(phase)

    phase_name = ["COND", "EXT", "AWARE"][phase]
    print(f"EP {ep+1:3d} [{phase_name}] | Ret={total:.2f} | Emo={emo_sum:.2f} | visits={visit_count} | eps={eps:.3f}")

# SAVE DATA
pickle.dump(
    {
        "returns": np.array(returns, dtype=np.float32),
        "emos": np.array(emos, dtype=np.float32),
        "visits": np.array(fear_visits, dtype=np.float32),
        "phases": np.array(phases, dtype=np.int32),
        "cfg": dict(COND_EP=COND_EP, EXT_EP=EXT_EP, AWARE_EP=AWARE_EP),
    },
    open("ex8_fear.pkl", "wb"),
)

print("SAVED: ex8_fear.pkl")