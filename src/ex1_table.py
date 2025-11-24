import pickle
import numpy as np

# Load saved experiment 1 data
with open("ex1_data.pkl", "rb") as f:
    data = pickle.load(f)

returns = np.array(data[0])   # return per episode
emos = np.array(data[1])      # emotional reward per episode

# Compute metrics
avg_return = np.mean(returns)
std_return = np.std(returns)
best_return = np.max(returns)
worst_return = np.min(returns)

# Failure definition: return < 0
failures = np.sum(returns < 0)
failure_rate = (failures / len(returns)) * 100

# Print summary table
print("\n====== EXPERIMENT 1 SUMMARY TABLE ======")
print(f"Total Episodes       : {len(returns)}")
print(f"Average Return       : {avg_return:.2f}")
print(f"Return Std Dev       : {std_return:.2f}")
print(f"Best Return          : {best_return:.2f}")
print(f"Worst Return         : {worst_return:.2f}")
print(f"Failure Episodes     : {failures}")
print(f"Failure Rate (%)     : {failure_rate:.2f}%")
print("========================================\n")