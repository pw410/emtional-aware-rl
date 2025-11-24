import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("ex1_data.pkl", "rb") as f:
    data = pickle.load(f)

returns = np.array(data[0])    # correct
emos = np.array(data[1])       # correct

# --- Failure rate ---
window = 20
failures = (returns < 0).astype(int)
fail_rate = np.convolve(failures, np.ones(window)/window, mode='same') * 100

plt.figure(figsize=(10,5))
plt.plot(fail_rate, color='red')
plt.title("Experiment 1: Failure Rate (Return < 0)")
plt.xlabel("Episode")
plt.ylabel("Failure Rate (%)")
plt.grid(True)
plt.savefig("ex1_final_failure.png")
plt.close()

# --- Return plot ---
plt.figure(figsize=(10,5))
plt.plot(returns, color='blue')
plt.title("Experiment 1: Return per Episode")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.savefig("ex1_final_return.png")
plt.close()

print("EX1 DONE ✓")