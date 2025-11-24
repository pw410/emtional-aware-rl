import pickle
import numpy as np
import matplotlib.pyplot as plt

base_trajs = pickle.load(open("ex10_traj_base.pkl", "rb"))
emo_trajs  = pickle.load(open("ex10_traj_emo.pkl", "rb"))

def summarize(trajs):
    totals = np.array([t["total_reward"] for t in trajs], dtype=np.float32)
    steps  = np.array([len(t["steps"]) for t in trajs], dtype=np.float32)
    lava   = np.array([sum(s["is_lava"] for s in t["steps"]) for t in trajs], dtype=np.float32)
    goal   = np.array([sum(s["is_goal"] for s in t["steps"]) for t in trajs], dtype=np.float32)
    distract = np.array([sum(s["is_distract"] for s in t["steps"]) for t in trajs], dtype=np.float32)
    return dict(
        total=totals,
        steps=steps,
        lava=lava,
        goal=goal,
        distract=distract,
    )

sb = summarize(base_trajs)
se = summarize(emo_trajs)

print("\n=== EXPERIMENT 10 SUMMARY (Trajectory-level) ===")

def print_stats(name, s):
    print(f"\n{name}:")
    print(f"  Avg Total Reward: {s['total'].mean():.2f}  (std {s['total'].std():.2f})")
    print(f"  Avg Steps:        {s['steps'].mean():.1f}")
    print(f"  Avg LAVA hits:    {s['lava'].mean():.2f}")
    print(f"  Avg GOAL reached: {s['goal'].mean():.2f}")
    print(f"  Avg DISTRACTOR visits: {s['distract'].mean():.2f}")

print_stats("Baseline", sb)
print_stats("Emotional", se)

# ---- Simple bar-plot comparison ----
labels = ["Total reward", "Lava hits", "Goal reached", "Distractor visits"]
baseline_vals = [
    sb["total"].mean(),
    sb["lava"].mean(),
    sb["goal"].mean(),
    sb["distract"].mean(),
]
emo_vals = [
    se["total"].mean(),
    se["lava"].mean(),
    se["goal"].mean(),
    se["distract"].mean(),
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, baseline_vals, width, label="Baseline")
plt.bar(x + width/2, emo_vals, width, label="Emotional")
plt.xticks(x, labels, rotation=20)
plt.ylabel("Average (per trajectory)")
plt.title("Experiment 10: Baseline vs Emotional Trajectory Stats")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("ex10_stats_bar.png")
plt.close()