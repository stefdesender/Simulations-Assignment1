import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Define policies
policies = {
    "Policy 0": {"P": np.array([[0.1, 0.9, 0.0, 0.0], [0.2, 0.0, 0.8, 0.0], [0.5, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]]), "cost": [600, 200, 500, 1000]},
    "Policy 1": {"P": np.array([[0.1, 0.9, 0.0], [0.2, 0.0, 0.8], [1.0, 0.0, 0.0]]), "cost": [600, 200, 500]},
    "Policy 2": {"P": np.array([[0.1, 0.9], [1.0, 0.0]]), "cost": [600, 200]},
    "Policy 3": {"P": np.array([[1.0]]), "cost": [600]}
}

number_periods = 1000
K = 20
warmup_period = 100
np.random.seed(42) # For reproducible results

# Matrix to store the exact cost incurred IN EACH MONTH
# Shape: (K runs, N periods)
raw_costs = {p: np.zeros((K, number_periods)) for p in policies}

# --- 1. SIMULATION LOOP ---
for p_name, data in policies.items():
    P = data["P"]
    cost = data["cost"]
    nr_states = len(P)
    F = np.cumsum(P, axis=1)
    F_s0 = np.zeros(nr_states)
    F_s0[0] = 1.0

    for k in range(K):
        # Initial state
        r = np.random.rand()
        i2 = 0
        while i2 < nr_states - 1 and r > F_s0[i2]: i2 += 1
        current_state = i2
        
        for i1 in range(1, number_periods + 1):
            # Next state
            r = np.random.rand()
            next_state = 0
            while next_state < nr_states - 1 and r > F[current_state][next_state]:
                next_state += 1
            current_state = next_state
            
            # Record cost
            raw_costs[p_name][k, i1-1] = cost[current_state]

# --- 2. CALCULATION AND PLOTTING ---
for idx, p_name in enumerate(policies):
    data_raw = raw_costs[p_name] 
    
    # --- WITHOUT WARMUP ---
    run_avg_std = np.cumsum(data_raw, axis=1) / np.arange(1, number_periods + 1)
    mean_std = np.mean(run_avg_std, axis=0)
    se_std = st.sem(run_avg_std, axis=0)
    ci_std = se_std * st.t.ppf((1 + 0.95) / 2., K-1)
    periods_std = np.arange(1, number_periods + 1)
    
    fig, ax_std = plt.subplots(figsize=(8, 6))
    ax_std.plot(periods_std, mean_std, label='Mean', color='tab:blue', linewidth=2)
    
    if p_name != "Policy 3":
        ax_std.plot(periods_std, mean_std + ci_std, label='CI+', color='tab:red', linestyle='--', linewidth=1)
        ax_std.plot(periods_std, mean_std - ci_std, label='CI-', color='tab:green', linestyle='--', linewidth=1)
        ax_std.fill_between(periods_std, mean_std - ci_std, mean_std + ci_std, color='gray', alpha=0.2)
    
    ax_std.set_title(f'{p_name}: Running Average (NO Warm-up Removal)', fontsize=14)
    ax_std.set_xlabel('Month', fontsize=12)
    ax_std.set_ylabel('Cost ($)', fontsize=12)
    ax_std.grid(True, linestyle='--', alpha=0.6)
    
    # Calculate limits for visual consistency between the two graphs
    if p_name != "Policy 3":
        data_warmup = data_raw[:, warmup_period:] 
        run_avg_wu = np.cumsum(data_warmup, axis=1) / np.arange(1, number_periods - warmup_period + 1)
        mean_wu = np.mean(run_avg_wu, axis=0)
        se_wu = st.sem(run_avg_wu, axis=0)
        ci_wu = se_wu * st.t.ppf((1 + 0.95) / 2., K-1)
        
        y_min = min(np.min(mean_std - ci_std), np.min(mean_wu - ci_wu)) - 20
        y_max = max(np.max(mean_std + ci_std), np.max(mean_wu + ci_wu)) + 20
        ax_std.set_ylim(y_min, y_max)
        ax_std.legend()
    else:
        ax_std.set_ylim(550, 650)
        
    plt.tight_layout()
    filename_no_wu = f"Graph_{idx+1}_{p_name.replace(' ', '_')}_NO_warmup.png"
    plt.savefig(filename_no_wu)
    plt.close() # Close figure to free memory

    # --- WITH WARMUP REMOVAL ---
    fig, ax_wu = plt.subplots(figsize=(8, 6))
    
    if p_name != "Policy 3":
        periods_wu = np.arange(warmup_period + 1, number_periods + 1)
        ax_wu.plot(periods_wu, mean_wu, label='Mean', color='tab:blue', linewidth=2)
        ax_wu.plot(periods_wu, mean_wu + ci_wu, label='CI+', color='tab:red', linestyle='--', linewidth=1)
        ax_wu.plot(periods_wu, mean_wu - ci_wu, label='CI-', color='tab:green', linestyle='--', linewidth=1)
        ax_wu.fill_between(periods_wu, mean_wu - ci_wu, mean_wu + ci_wu, color='gray', alpha=0.2)
        ax_wu.set_ylim(y_min, y_max)
        ax_wu.legend()
    else:
        periods_wu = np.arange(warmup_period + 1, number_periods + 1)
        data_warmup = data_raw[:, warmup_period:]
        run_avg_wu = np.cumsum(data_warmup, axis=1) / np.arange(1, number_periods - warmup_period + 1)
        mean_wu = np.mean(run_avg_wu, axis=0)
        ax_wu.plot(periods_wu, mean_wu, label='Mean', color='tab:blue', linewidth=2)
        ax_wu.set_ylim(550, 650)
    
    ax_wu.set_title(f'{p_name}: Running Average (WITH Warm-up Removal > {warmup_period})', fontsize=14)
    ax_wu.set_xlabel('Month', fontsize=12)
    ax_wu.set_ylabel('Cost ($)', fontsize=12)
    ax_wu.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    filename_wu = f"Graph_{idx+1}_{p_name.replace(' ', '_')}_WITH_warmup.png"
    plt.savefig(filename_wu)
    plt.close()

print("All 8 graphs have been successfully generated and saved!")