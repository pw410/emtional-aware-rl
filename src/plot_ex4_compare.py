import pickle
import numpy as np
import matplotlib.pyplot as plt

# load both conditions
ret_off, emo_off, fail_off = pickle.load(open("ex4_mirror_off.pkl", "rb"))
ret_on,  emo_on,  fail_on  = pickle.load(open("ex4_mirror_on.pkl", "rb"))

episodes = np.arange(1, len(ret_off)+1)

# ---------- G1: Return vs Episode ----------
plt.figure(figsize=(10,5))
plt.plot(episodes, ret_off, label="Mirror OFF")
plt.plot(episodes, ret_on,  label="Mirror ON")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Experiment 4: Return vs Episode (Mirror Ablation)")
plt.legend()
plt.grid(True)
plt.savefig("ex4_return_compare.png")
plt.close()

# ---------- G2: Emotional Reward vs Episode (ON only) ----------
plt.figure(figsize=(10,5))
plt.plot(episodes, emo_on, color="orange")
plt.xlabel("Episode")
plt.ylabel("Total Emo Reward")
plt.title("Experiment 4: Emotional Reward (Mirror ON)")
plt.grid(True)
plt.savefig("ex4_emo_on.png")
plt.close()

# ---------- G3: Failure rate moving average ----------
window = 20
def moving_avg(x, w):
    x = np.array(x, dtype=np.float32)
    return np.convolve(x, np.ones(w)/w, mode="valid")

ma_off = moving_avg(fail_off, window)
ma_on  = moving_avg(fail_on,  window)

ep_ma = np.arange(1, len(ma_off)+1)

plt.figure(figsize=(10,5))
plt.plot(ep_ma, ma_off*100, label="Mirror OFF")
plt.plot(ep_ma, ma_on*100,  label="Mirror ON")
plt.xlabel("Episode")
plt.ylabel("Failure rate (%)")
plt.title(f"Experiment 4: Lava Failure Rate (window={window})")
plt.legend()
plt.grid(True)
plt.savefig("ex4_failure_compare.png")
plt.close()

# ---------- TABLE ----------
print("\n=== EXPERIMENT 4 SUMMARY ===")
print(f"Mirror OFF  Avg Return: {np.mean(ret_off):.2f}  Std: {np.std(ret_off):.2f}")
print(f"Mirror ON   Avg Return: {np.mean(ret_on):.2f}   Std: {np.std(ret_on):.2f}")
print(f"Mirror OFF  Failure Rate: {np.mean(fail_off)*100:.1f}%")
print(f"Mirror ON   Failure Rate: {np.mean(fail_on)*100:.1f}%")
print(f"Mirror OFF  Avg Emo: {np.mean(emo_off):.2f}")
print(f"Mirror ON   Avg Emo: {np.mean(emo_on):.2f}")