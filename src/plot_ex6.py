import pickle
import numpy as np
import matplotlib.pyplot as plt

NOISE_LEVELS = [0.0, 0.5, 1.0]

def load_pair(std):
    r_b, e_b, f_b = pickle.load(open(f"ex6_noise{std}_base.pkl", "rb"))
    r_e, e_e, f_e = pickle.load(open(f"ex6_noise{std}_emo.pkl", "rb"))
    return (np.array(r_b), np.array(e_b), np.array(f_b)), (np.array(r_e), np.array(e_e), np.array(f_e))

# ---- per-noise plots + summary stats ----
avg_ret_base = []
avg_ret_emo = []
fail_base = []
fail_emo = []

for nl in NOISE_LEVELS:
    (r_b, e_b, f_b), (r_e, e_e, f_e) = load_pair(nl)
    episodes = np.arange(1, len(r_b)+1)

    # G1: Return vs Episode
    plt.figure(figsize=(10,5))
    plt.plot(episodes, r_b, label="Baseline")
    plt.plot(episodes, r_e, label="Emotional")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Experiment 6 (noise={nl}) - Return vs Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ex6_noise{nl}_return.png")
    plt.close()

    # collect stats
    avg_ret_base.append(r_b.mean())
    avg_ret_emo.append(r_e.mean())
    fail_base.append(f_b.mean() * 100.0)
    fail_emo.append(f_e.mean() * 100.0)

# ---- summary plot: Avg return vs noise ----
plt.figure(figsize=(8,5))
plt.plot(NOISE_LEVELS, avg_ret_base, marker="o", label="Baseline")
plt.plot(NOISE_LEVELS, avg_ret_emo, marker="o", label="Emotional")
plt.xlabel("Reward noise std")
plt.ylabel("Average Return")
plt.title("Experiment 6 - Avg Return vs Noise")
plt.legend()
plt.grid(True)
plt.savefig("ex6_summary_return.png")
plt.close()

# ---- summary plot: Failure rate vs noise ----
plt.figure(figsize=(8,5))
plt.plot(NOISE_LEVELS, fail_base, marker="o", label="Baseline")
plt.plot(NOISE_LEVELS, fail_emo, marker="o", label="Emotional")
plt.xlabel("Reward noise std")
plt.ylabel("Failure rate (%)")
plt.title("Experiment 6 - Failure Rate vs Noise")
plt.legend()
plt.grid(True)
plt.savefig("ex6_summary_fail.png")
plt.close()

# ---- TEXT TABLE ----
print("\n=== EXPERIMENT 6 SUMMARY ===")
for i, nl in enumerate(NOISE_LEVELS):
    print(f"Noise std = {nl}")
    print(f"  Baseline  AvgReturn={avg_ret_base[i]:.2f}  Fail%={fail_base[i]:.1f}")
    print(f"  Emotional AvgReturn={avg_ret_emo[i]:.2f}  Fail%={fail_emo[i]:.1f}")