import pickle
import matplotlib.pyplot as plt
import numpy as np

# load data
returns, emos = pickle.load(open("ex1_data.pkl", "rb"))

episodes = np.arange(1, len(returns) + 1)

# -------- PLOT 1: Return vs Episode --------
plt.figure(figsize=(10,5))
plt.plot(episodes, returns)
plt.title("Experiment 1: Return vs Episode")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.savefig("ex1_return_plot.png")
plt.close()

# -------- PLOT 2: Emotional Reward vs Episode --------
plt.figure(figsize=(10,5))
plt.plot(episodes, emos, color='orange')
plt.title("Experiment 1: Emotional Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Total Emotional Reward")
plt.grid(True)
plt.savefig("ex1_emo_plot.png")
plt.close()

# -------- TABLE SUMMARY --------
print("\n=== Experiment 1 Summary ===")
print(f"Average Return: {np.mean(returns):.2f}")
print(f"Return Std Dev: {np.std(returns):.2f}")
print(f"Average Emo Reward: {np.mean(emos):.2f}")
print(f"Emo Reward Std Dev: {np.std(emos):.2f}")