import pickle
import numpy as np
import matplotlib.pyplot as plt

# load
base_ret, base_fail, base_emo = pickle.load(open("ex7_base.pkl", "rb"))
emo_ret, emo_fail, emo_emo = pickle.load(open("ex7_emo.pkl", "rb"))

episodes = np.arange(1, base_ret.shape[1] + 1)

def mean_sem(x):
    m = x.mean(axis=0)
    s = x.std(axis=0) / np.sqrt(x.shape[0])
    return m, s

# --------- 1) Return vs Episode (mean ± sem) ----------
m_b, se_b = mean_sem(base_ret)
m_e, se_e = mean_sem(emo_ret)

plt.figure(figsize=(10,5))
plt.plot(episodes, m_b, label="Baseline")
plt.plot(episodes, m_e, label="Emotional")
plt.fill_between(episodes, m_b - se_b, m_b + se_b, alpha=0.2)
plt.fill_between(episodes, m_e - se_e, m_e + se_e, alpha=0.2)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Experiment 7: Sample Efficiency - Avg Return vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("ex7_return_curve.png")
plt.close()

# --------- 2) Safety curve: failure prob vs episode ----------
m_fb, _ = mean_sem(base_fail)
m_fe, _ = mean_sem(emo_fail)

plt.figure(figsize=(10,5))
plt.plot(episodes, m_fb * 100, label="Baseline")
plt.plot(episodes, m_fe * 100, label="Emotional")
plt.xlabel("Episode")
plt.ylabel("Failure probability (%)")
plt.title("Experiment 7: Safety Curve - Failure vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("ex7_safety_curve.png")
plt.close()

# --------- 3) Emotional magnitude over learning ----------
m_emo, _ = mean_sem(emo_emo)
plt.figure(figsize=(10,5))
plt.plot(episodes, m_emo)
plt.xlabel("Episode")
plt.ylabel("Total Emotional Reward")
plt.title("Experiment 7: Emotional Dynamics vs Episode (Emotional agent)")
plt.grid(True)
plt.savefig("ex7_emo_curve.png")
plt.close()

# --------- 4) TABLE SUMMARY ----------
print("\n=== EXPERIMENT 7 SUMMARY ===")
print(f"Baseline  Avg Return: {base_ret.mean():.2f}")
print(f"Emotional Avg Return: {emo_ret.mean():.2f}")
print(f"Baseline  Avg Failure%: {base_fail.mean()*100:.2f}")
print(f"Emotional Avg Failure%: {emo_fail.mean()*100:.2f}")
print(f"Emotional Avg Emo Reward per ep: {emo_emo.mean():.2f}")