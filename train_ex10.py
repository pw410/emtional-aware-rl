from grid_env_multi import DistractGrid11
from dqn_agent import DQNAgent
from emotion_module_ex5 import EmotionModuleEx5

import torch
import numpy as np
import pickle

EPISODES = 300
MAX_STEPS = 200
LAMBDA_EMO = 2.0
N_TRAJ = 10   # how many trajectories per agent to save


def train_agent(use_emotion=True):
    env = DistractGrid11()
    agent = DQNAgent(state_dim=6, action_dim=4)
    emo = EmotionModuleEx5() if use_emotion else None

    eps = 1.0
    for ep in range(EPISODES):
        s = env.reset()
        total = 0.0
        emo_sum = 0.0

        for t in range(MAX_STEPS):
            a = agent.act(s, eps)
            ns, R_ext, done = env.step(a)

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
        tag = "EMO" if use_emotion else "BASE"
        print(f"{tag} TRAIN | EP {ep+1} | Ret={total:.2f} | Emo={emo_sum:.2f} | eps={eps:.3f}")

    return agent, emo


def record_trajectories(agent, emo, use_emotion=True, n_traj=10, max_steps=200):
    env = DistractGrid11()
    trajs = []

    for k in range(n_traj):
        s = env.reset()
        eps_eval = 0.0   # greedy policy for evaluation

        steps = []
        total_R_ext = 0.0
        total_R_emo = 0.0
        total_R = 0.0

        for t in range(max_steps):
            a = agent.act(s, eps_eval)
            ns, R_ext, done = env.step(a)

            ax, ay = env.agent[0], env.agent[1]
            lx, ly = env.lava
            gx, gy = env.goal
            is_lava = (ax, ay) == (lx, ly)
            is_goal = (ax, ay) == (gx, gy)
            is_distract = (ax, ay) in getattr(env, "distractors", set())

            if use_emotion:
                R_emo = emo(torch.tensor([s], dtype=torch.float32)).item()
                R = R_ext + LAMBDA_EMO * R_emo
            else:
                R_emo = 0.0
                R = R_ext

            step_info = dict(
                t=t,
                state=s.tolist(),
                action=int(a),
                ext_reward=float(R_ext),
                emo_reward=float(R_emo),
                total_reward=float(R),
                pos=(int(ax), int(ay)),
                is_lava=bool(is_lava),
                is_goal=bool(is_goal),
                is_distract=bool(is_distract),
                done=bool(done),
            )
            steps.append(step_info)

            total_R_ext += R_ext
            total_R_emo += R_emo
            total_R += R

            s = ns
            if done:
                break

        trajs.append(
            dict(
                steps=steps,
                total_ext_reward=float(total_R_ext),
                total_emo_reward=float(total_R_emo),
                total_reward=float(total_R),
            )
        )
        tag = "EMO" if use_emotion else "BASE"
        print(f"{tag} TRAJ {k+1}/{n_traj} | steps={len(steps)} | totalR={total_R:.2f}")

    return trajs


def write_human_readable(trajs_base, trajs_emo, filename="ex10_trajectories.txt"):
    lines = []
    lines.append("EXPERIMENT 10 - HUMAN EVALUATION TRAJECTORIES\n")
    lines.append("Environment: DistractGrid11 (goal, lava, 3 distractor cells)\n")
    lines.append("Each trajectory = list of (step, position, action, rewards, event)\n")
    lines.append("="*80 + "\n\n")

    def describe_step(s):
        ev = []
        if s["is_goal"]:
            ev.append("GOAL")
        if s["is_lava"]:
            ev.append("LAVA")
        if s["is_distract"]:
            ev.append("DISTRACTOR")
        event = ",".join(ev) if ev else "-"
        return f"t={s['t']:02d}  pos={s['pos']}  a={s['action']}  ext={s['ext_reward']:+.2f}  emo={s['emo_reward']:+.2f}  tot={s['total_reward']:+.2f}  event={event}"

    # Baseline
    lines.append("==== BASELINE AGENT TRAJECTORIES ====\n\n")
    for i, traj in enumerate(trajs_base):
        lines.append(f"[BASE TRAJ {i+1}]  steps={len(traj['steps'])}  total_ext={traj['total_ext_reward']:.2f}  total_emo={traj['total_emo_reward']:.2f}  total={traj['total_reward']:.2f}\n")
        for s in traj["steps"]:
            lines.append(describe_step(s) + "\n")
        lines.append("\n")

    # Emotional
    lines.append("\n==== EMOTIONAL AGENT TRAJECTORIES ====\n\n")
    for i, traj in enumerate(trajs_emo):
        lines.append(f"[EMO TRAJ {i+1}]  steps={len(traj['steps'])}  total_ext={traj['total_ext_reward']:.2f}  total_emo={traj['total_emo_reward']:.2f}  total={traj['total_reward']:.2f}\n")
        for s in traj["steps"]:
            lines.append(describe_step(s) + "\n")
        lines.append("\n")

    with open(filename, "w") as f:
        f.writelines(lines)
    print("WROTE HUMAN-READABLE FILE:", filename)


if __name__ == "__main__":
    # 1) Train baseline
    print("=== TRAINING BASELINE AGENT (no emotion) ===")
    base_agent, _ = train_agent(use_emotion=False)

    # 2) Train emotional agent
    print("\n=== TRAINING EMOTIONAL AGENT ===")
    emo_agent, emo_module = train_agent(use_emotion=True)

    # 3) Record trajectories
    print("\n=== RECORDING TRAJECTORIES FOR HUMAN EVALUATION ===")
    trajs_base = record_trajectories(base_agent, emo_module, use_emotion=False, n_traj=N_TRAJ, max_steps=MAX_STEPS)
    trajs_emo = record_trajectories(emo_agent, emo_module, use_emotion=True, n_traj=N_TRAJ, max_steps=MAX_STEPS)

    # 4) Save pickles
    pickle.dump(trajs_base, open("ex10_traj_base.pkl", "wb"))
    pickle.dump(trajs_emo, open("ex10_traj_emo.pkl", "wb"))
    print("SAVED: ex10_traj_base.pkl, ex10_traj_emo.pkl")

    # 5) Human-readable text file
    write_human_readable(trajs_base, trajs_emo, filename="ex10_trajectories.txt")