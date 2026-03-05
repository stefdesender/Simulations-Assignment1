import matplotlib.pyplot as plt
from scipy.stats import bernoulli

K = 10000
max_tosses = 6
total_success = 0

k_values = []
running_averages = []

print("Running 10,000 simulations using scipy.stats.bernoulli...")

for k in range(1, K + 1):
    count_success = 0
    nr_tosses = 0
    
    # First toss using scipy.stats
    current_toss = bernoulli.rvs(0.5)
    nr_tosses += 1
    
    # Subsequent tosses
    while nr_tosses < max_tosses:
        toss = bernoulli.rvs(0.5)
        previous_toss = current_toss
        current_toss = toss
        nr_tosses += 1
        
        # Heads should be followed by tails and vice versa
        if previous_toss != current_toss:
            count_success += 1
            
    # If experiment = success (5 alternating shifts in 6 tosses)
    if count_success == max_tosses - 1:
        total_success += 1
        
    # Keep track of the running average for plotting
    p_hat_k = total_success / k
    k_values.append(k)
    running_averages.append(p_hat_k)

# --- 3. CALCULATE PROBABILITY & CONFIDENCE INTERVAL ---
p_hat = total_success / K
Z = 1.96  # Z-score for a 95% confidence level

# Manual calculation of standard error and margin of error
standard_error = (p_hat * (1 - p_hat) / K) ** 0.5
margin_of_error = Z * standard_error

lower_bound = p_hat - margin_of_error
upper_bound = p_hat + margin_of_error

# --- 4. OUTPUT RESULTS ---
print("-" * 40)
print(f"Total successes:         {total_success}")
print(f"Simulated Probability:   {p_hat:.5f}")
print(f"Theoretical Probability: 0.03125")
print("-" * 40)
print(f"95% Confidence Interval: [{lower_bound:.5f}, {upper_bound:.5f}]")
print(f"Margin of Error:         ±{margin_of_error:.5f}")

# --- 5. PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot running average
plt.plot(k_values, running_averages, label='Simulated Probability', color='purple', linewidth=1.5)

# Plot theoretical probability
plt.axhline(y=0.03125, color='orange', linestyle='--', label='Theoretical Probability (0.03125)')

# Highlight the final confidence interval area
plt.fill_between([1, K], lower_bound, upper_bound, color='gray', alpha=0.3, label='Final 95% Confidence Interval')

plt.title('Convergence of Simulated Probability (Using scipy.stats.bernoulli)')
plt.xlabel('Number of Trials (k)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('scipy_bernoulli_convergence.png', bbox_inches='tight')
plt.show()