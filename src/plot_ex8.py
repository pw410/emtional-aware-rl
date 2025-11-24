import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open("ex8_fear.pkl", "rb"))

returns = data["returns"]
emos = data["emos"]
visits = data["visits"]
phases = data["phases"]
cfg = data["cfg"]

COND_EP = cfg["COND_EP"]
EXT_EP = cfg["EXT_EP"]
AWARE_EP = cfg["AWARE_EP"]

episodes = np.arange(1, len(returns) + 1)

# ---------- helper to slice by phase ----------
def phase_slice(phase_id):
    idx = np.where(phases == phase_id)[0]
    return idx, returns[idx], emos[idx], visits[idx]

# ---------- 1) Return vs Episode ----------
plt.figure(figsize=(10,5))
plt.plot(episodes, returns)
plt.axvline(COND_EP, color="red", linestyle="--", label="Cond → Extinction")
plt.axvline(COND_EP + EXT_EP, color="green", linestyle="--", label="Extinction → Awareness")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Experiment 8: Fear Conditioning / Extinction - Return vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("ex8_return.png")
plt.close()

# ---------- 2) Fear-tile visits vs Episode ----------
plt.figure(figsize=(10,5))
plt.plot(episodes, visits)
plt.axvline(COND_EP, color="red", linestyle="--")
plt.axvline(COND_EP + EXT_EP, color="green", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Visits to Fear Tile")
plt.title("Experiment 8: Fear Tile Visits per Episode")
plt.grid(True)
plt.savefig("ex8_visits.png")
plt.close()

# ---------- 3) Emotional reward vs Episode ----------
plt.figure(figsize=(10,5))
plt.plot(episodes, emos)
plt.axvline(COND_EP, color="red", linestyle="--")
plt.axvline(COND_EP + EXT_EP, color="green", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Total Emotional Reward")
plt.title("Experiment 8: Emotional Reward vs Episode")
plt.grid(True)
plt.savefig("ex8_emo.png")
plt.close()

# ---------- 4) TABLE SUMMARY ----------
print("\n=== EXPERIMENT 8 SUMMARY (per phase) ===")

names = {0: "Conditioning", 1: "Extinction", 2: "Awareness"}
for pid in [0, 1, 2]:
    idx, r, e, v = phase_slice(pid)
    print(f"\nPhase {pid} ({names[pid]}): EP {idx[0]+1}-{idx[-1]+1}")
    print(f"  Avg Return: {r.mean():.2f}")
    print(f"  Avg Emo:    {e.mean():.2f}")
    print(f"  Avg Visits: {v.mean():.2f}")