import pickle
import numpy as np
import matplotlib.pyplot as plt

returns, emos, avoid = pickle.load(open("ex3_data.pkl","rb"))

episodes = np.arange(1, len(returns)+1)

# G1: Return
plt.figure(figsize=(10,5))
plt.plot(episodes, returns)
plt.title("Experiment 3: Return vs Episode")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid()
plt.savefig("ex3_return.png")
plt.close()

# G2: Emotional Reward
plt.figure(figsize=(10,5))
plt.plot(episodes, emos, color='orange')
plt.title("Emotional Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Total Emo")
plt.grid()
plt.savefig("ex3_emo.png")
plt.close()

# G3: Fear Avoidance Curve
plt.figure(figsize=(10,5))
plt.plot(episodes, avoid, color='red')
plt.title("Fear Avoidance (Lava Avoid) Curve")
plt.xlabel("Episode")
plt.ylabel("Avoidance Score")
plt.grid()
plt.savefig("ex3_avoid.png")
plt.close()

# TABLE
print("\n=== EXPERIMENT 3 SUMMARY ===")
print(f"Avg Return: {np.mean(returns):.2f}")
print(f"Avg Emo: {np.mean(emos):.2f}")
print(f"Avg Avoidance: {np.mean(avoid):.2f}")