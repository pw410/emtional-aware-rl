import pickle
import numpy as np
import matplotlib.pyplot as plt

TASKS = ["risky", "delayed", "distract"]

def load_pair(task):
    r_b, e_b, f_b = pickle.load(open(f"ex5_{task}_base.pkl", "rb"))
    r_e, e_e, f_e = pickle.load(open(f"ex5_{task}_emo.pkl", "rb"))
    return (np.array(r_b), np.array(e_b), np.array(f_b)), (np.array(r_e), np.array(e_e), np.array(f_e))

# --------- PLOTS ----------
for task in TASKS:
    (r_b, e_b, f_b), (r_e, e_e, f_e) = load_pair(task)
    episodes = np.arange(1, len(r_b)+1)

    # Return vs episode
    plt.figure(figsize=(10,5))
    plt.plot(episodes, r_b, label="Baseline")
    plt.plot(episodes, r_e, label="Emotional")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Experiment 5 ({task}) - Return vs Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ex5_{task}_return.png")
    plt.close()

    # Failure moving average
    window = 20
    def ma(x,w):
        return np.convolve(x, np.ones(w)/w, mode="valid")
    fb_ma = ma(f_b, window)
    fe_ma = ma(f_e, window)
    ep_ma = np.arange(1, len(fb_ma)+1)

    plt.figure(figsize=(10,5))
    plt.plot(ep_ma, fb_ma*100, label="Baseline")
    plt.plot(ep_ma, fe_ma*100, label="Emotional")
    plt.xlabel("Episode")
    plt.ylabel("Failure rate (%)")
    plt.title(f"Experiment 5 ({task}) - Failure Rate (window={window})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ex5_{task}_fail.png")
    plt.close()

# --------- TABLE SUMMARY ----------
print("\n=== EXPERIMENT 5 SUMMARY ===")
for task in TASKS:
    (r_b, e_b, f_b), (r_e, e_e, f_e) = load_pair(task)
    print(f"\nTask: {task}")
    print(f"  Baseline  Avg Return: {r_b.mean():.2f}  Std: {r_b.std():.2f}  Fail%: {f_b.mean()*100:.1f}")
    print(f"  Emotional Avg Return: {r_e.mean():.2f}  Std: {r_e.std():.2f}  Fail%: {f_e.mean()*100:.1f}")