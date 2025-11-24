import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD EXPERIMENT 2 DATA
# -----------------------------
base = pickle.load(open("ex2_base.pkl", "rb"))
emo = pickle.load(open("ex2_emo.pkl", "rb"))
aware = pickle.load(open("ex2_aware.pkl", "rb"))
both = pickle.load(open("ex2_emo_aware.pkl", "rb"))

# Convert to numpy
base = np.array(base)
emo = np.array(emo)
aware = np.array(aware)
both = np.array(both)

# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_curve(data, title, filename):
    plt.figure(figsize=(8,5))
    plt.plot(data, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -----------------------------
#   CREATE ALL 4 PLOTS
# -----------------------------
plot_curve(base,  "Experiment 2: Baseline (No Emotion, No Awareness)", "ex2_plot_baseline.png")
plot_curve(emo,   "Experiment 2: Emotion Only",                       "ex2_plot_emotion.png")
plot_curve(aware, "Experiment 2: Awareness Only",                     "ex2_plot_awareness.png")
plot_curve(both,  "Experiment 2: Emotion + Awareness",                "ex2_plot_both.png")

print("Graphs Saved: ex2_plot_baseline.png, ex2_plot_emotion.png, ex2_plot_awareness.png, ex2_plot_both.png")

# -----------------------------
#  COMPARISON TABLE
# -----------------------------
print("\n=================== EXPERIMENT 2 RESULTS ===================")
print("Model\t\tFinal Return\tAverage Return")
print("------------------------------------------------------------")
print(f"Baseline:\t{base[-1]:.2f}\t\t{np.mean(base):.2f}")
print(f"Emotion:\t{emo[-1]:.2f}\t\t{np.mean(emo):.2f}")
print(f"Awareness:\t{aware[-1]:.2f}\t\t{np.mean(aware):.2f}")
print(f"Both:\t\t{both[-1]:.2f}\t\t{np.mean(both):.2f}")
print("============================================================")