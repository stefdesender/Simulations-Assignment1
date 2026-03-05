import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np
import pandas as pd

np.random.seed(10)

K = 10000
max_tosses = 6
total_success = 0

k_values = []
running_averages = []
simulation_data = []

print("Running 10,000 simulations ")

for k in range(1, K + 1):
    count_success = 0
    nr_tosses = 0
    

    current_toss = bernoulli.rvs(0.5)
    nr_tosses += 1
    
    
    while nr_tosses < max_tosses:
        toss = bernoulli.rvs(0.5)
        previous_toss = current_toss
        current_toss = toss
        nr_tosses += 1
        
        # Heads should be followed by tails and vice versa
        if previous_toss != current_toss:
            count_success += 1
            
    
    if count_success == max_tosses - 1:
        total_success += 1
        
    is_success = 1 if (count_success == max_tosses - 1) else 0

    p_hat_k = total_success / k
    k_values.append(k)
    running_averages.append(p_hat_k)

    # Calculate running probability
    running_avg = total_success / k

    simulation_data.append({
        "Trial (k)": k,
        "Trial Result (1=Success, 0=Fail)": is_success,
        "Cumulative Successes": total_success,
        "Running Probability": running_avg
    })


df = pd.DataFrame(simulation_data)

# Export the DataFrame to an Excel file
excel_filename = "coin_toss_simulation.xlsx"
df.to_excel(excel_filename, index=False)


p_hat = total_success / K
Z = 1.96  # Z-score for a 95% confidence level


standard_error = (p_hat * (1 - p_hat) / K) ** 0.5
margin_of_error = Z * standard_error

lower_bound = p_hat - margin_of_error
upper_bound = p_hat + margin_of_error



#OUTPUT RESULTS
print("-" * 40)
print(f"Total successes:         {total_success}")
print(f"Simulated Probability:   {p_hat:.5f}")
print(f"Theoretical Probability: 0.03125")
print("-" * 40)
print(f"95% Confidence Interval: [{lower_bound:.5f}, {upper_bound:.5f}]")
print(f"Margin of Error:         ±{margin_of_error:.5f}")


plt.figure(figsize=(10, 6))


plt.plot(k_values, running_averages, label='Simulated Probability', color='purple', linewidth=1.5)

# Plot theoretical probability
plt.axhline(y=0.03125, color='orange', linestyle='--', label='Theoretical Probability (0.03125)')

plt.fill_between([1, K], lower_bound, upper_bound, color='gray', alpha=0.3, label='Final 95% Confidence Interval')

plt.title('Convergence of Simulated Probability')
plt.xlabel('Number of Trials (k)')
plt.ylabel('Probability', )
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('scipy_bernoulli_convergence.png', bbox_inches='tight')
plt.show()