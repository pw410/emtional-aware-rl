import pickle
import numpy as np
import matplotlib.pyplot as plt

COND_NAMES = [
    "mirror_off_aware",
    "mirror_on_noaware",
    "mirror_on_aware",
]

LABELS = {
    "mirror_off_aware":  "Mirror OFF (control)",
    "mirror_on_noaware": "Mirror ON, Awareness OFF",
    "mirror_on_aware":   "Mirror ON, Awareness ON",
}

colors = {
    "mirror_off_aware":  "blue",
    "mirror_on_noaware": "red",
    "mirror_on_aware":   "green",
}

def load(name):
    data = pickle.load(open(f"ex9_{name}.pkl", "rb"))
    return data

def moving_avg(x, w):
    x = np.array(x, dtype=np.float32)
    return np.convolve(x, np.ones(w)/w, mode="valid")

# ---------- 1) Return vs Episode ----------
plt.figure(figsize=(10,5))
for name in COND_NAMES:
    d = load(name)
    R = d["returns"]
    eps = np.arange(1, len(R)+1)
    plt.plot(eps, R, label=LABELS[name])
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Experiment 9: Return vs Episode (Empathy & Awareness)")
plt.legend()
plt.grid(True)
plt.savefig("ex9_return.png")
plt.close()

# ---------- 2) Failure rate moving average ----------
window = 20
plt.figure(figsize=(10,5))
for name in COND_NAMES:
    d = load(name)
    f = d["fails"]
    ma = moving_avg(f, window) * 100.0
    eps = np.arange(1, len(ma)+1)
    plt.plot(eps, ma, label=LABELS[name])
plt.xlabel("Episode")
plt.ylabel("Failure rate (%)")
plt.title(f"Experiment 9: Self Failure vs Episode (window={window})")
plt.legend()
plt.grid(True)
plt.savefig("ex9_fail.png")
plt.close()

# ---------- 3) Contagion rate moving average ----------
plt.figure(figsize=(10,5))
for name in COND_NAMES:
    d = load(name)
    c = d["contagion"]
    ma = moving_avg(c, window) * 100.0
    eps = np.arange(1, len(ma)+1)
    plt.plot(eps, ma, label=LABELS[name])
plt.xlabel("Episode")
plt.ylabel("Contagion rate (%)")
plt.title(f"Experiment 9: Emotional Contagion (other hurt → self fail) (window={window})")
plt.legend()
plt.grid(True)
plt.savefig("ex9_contagion.png")
plt.close()

# ---------- 4) TABLE SUMMARY ----------
print("\n=== EXPERIMENT 9 SUMMARY ===")
for name in COND_NAMES:
    d = load(name)
    R = d["returns"]
    F = d["fails"]
    C = d["contagion"]
    E = d["emos"]
    print(f"\nCondition: {LABELS[name]}")
    print(f"  Avg Return: {R.mean():.2f}")
    print(f"  Avg Failure%: {F.mean()*100:.2f}")
    print(f"  Avg Contagion%: {C.mean()*100:.2f}")
    print(f"  Avg Emo Reward per ep: {E.mean():.2f}")