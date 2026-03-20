import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

# Policies
policies = {
    "Policy 0": {"P": np.array([[0.1, 0.9, 0.0, 0.0], [0.2, 0.0, 0.8, 0.0], [0.5, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]]), "cost": [600, 200, 500, 1000]},
    "Policy 1": {"P": np.array([[0.1, 0.9, 0.0], [0.2, 0.0, 0.8], [1.0, 0.0, 0.0]]), "cost": [600, 200, 500]},
    "Policy 2": {"P": np.array([[0.1, 0.9], [1.0, 0.0]]), "cost": [600, 200]},
    "Policy 3": {"P": np.array([[1.0]]), "cost": [600]}
}

number_periods = 1000
K = 20

# Setting a conservative, safe warm-up period based on visual inspection
warmup_period = 100 
window_size = 20 # Window for Welch's moving average smoothing
np.random.seed(42)

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
        r = np.random.rand()
        i2 = 0
        while i2 < nr_states - 1 and r > F_s0[i2]: i2 += 1
        current_state = i2
        
        for i1 in range(1, number_periods + 1):
            r = np.random.rand()
            next_state = 0
            while next_state < nr_states - 1 and r > F[current_state][next_state]:
                next_state += 1
            current_state = next_state
            
            raw_costs[p_name][k, i1-1] = cost[current_state]

results_std = []
results_wu = []
def welch_moving_average(series, w):
    """
    Welch moving average using 0-based indexing.

    Parameters
    ----------
    series : 1D array-like
        The time series to smooth
    w : int
        Welch window size

    Returns
    -------
    np.ndarray
        Smoothed series
    """
    series = np.asarray(series, dtype=float)
    T = len(series)
    smoothed = np.zeros(T)

    for t in range(T):   # t = 0, 1, ..., T-1
        if t < w:
            # beginning: use periods 0 ... 2t
            smoothed[t] = np.mean(series[0 : 2*t + 1])

        elif t <= T - w - 1:
            # middle: use periods t-w ... t+w
            smoothed[t] = np.mean(series[t - w : t + w + 1])

        else:
            # end: use symmetric window that fits, periods 2t-T+1 ... T-1
            start_idx = 2*t - T + 1
            smoothed[t] = np.mean(series[start_idx : T])

    return smoothed
# --- 2. CALCULATIONS & PLOTTING ---
for idx, p_name in enumerate(policies):
    data_raw = raw_costs[p_name] 
    
    # =========================================================
    # GRAPH A: MULTIPLE REPLICATION METHOD (Welch's Warm-up Check)
    # =========================================================
 # Plot both raw ensemble average and smoothed Welch curve
    ensemble_avg = np.mean(data_raw, axis=0)
    smoothed_avg = welch_moving_average(ensemble_avg, window_size)
    fig, ax_welch = plt.subplots(figsize=(8, 6))
    periods_all = np.arange(1, number_periods + 1)

    ax_welch.plot(
        periods_all,
        ensemble_avg,
        color="gray",
        alpha=0.35,
        linewidth=1.5,
        label=f"Raw ensemble average (K={K})"
    )

    ax_welch.plot(
        periods_all,
        smoothed_avg,
        color="tab:red",
        linewidth=2.5,
        label=f"Welch moving average (w={window_size})"
    )

    ax_welch.axvline(
        x=warmup_period,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Chosen warm-up t0={warmup_period}"
    )

    ax_welch.set_title(f"{p_name}: Welch Warm-up Analysis", fontsize=14)
    ax_welch.set_xlabel("Simulation Period (Months)", fontsize=12)
    ax_welch.set_ylabel("Average Cost ($)", fontsize=12)
    ax_welch.grid(True, linestyle="--", alpha=0.6)
    ax_welch.legend()
    plt.tight_layout()
    plt.savefig(f"Graph_{idx+1}A_{p_name.replace(' ', '_')}_Welch.png")
    plt.show()
    plt.close()

    # =========================================================
    # GRAPH B: WITHOUT WARMUP (Standard Running Average)
    # =========================================================
    # run_avg_std = np.cumsum(data_raw, axis=1) / np.arange(1, number_periods + 1)
    # mean_std = np.mean(run_avg_std, axis=0)
    # se_std = st.sem(run_avg_std, axis=0)
    # ci_std = se_std * st.t.ppf((1 + 0.95) / 2., K-1)
    
    # results_std.append({
    #     "Policy": p_name, 
    #     "Grand Mean Cost ($)": f"{mean_std[-1]:.2f}", 
    #     "95% CI (+/-)": f"{ci_std[-1]:.2f}",
    #     "95% CI Interval": f"[{mean_std[-1] - ci_std[-1]:.2f}, {mean_std[-1] + ci_std[-1]:.2f}]"
    # })
    
    # fig, ax_std = plt.subplots(figsize=(8, 6))
    # ax_std.plot(periods_all, mean_std, label='Grand Mean', color='tab:blue', linewidth=2)
    
    # if p_name != "Policy 3":
    #     ax_std.plot(periods_all, mean_std + ci_std, label='CI+', color='tab:red', linestyle='--', linewidth=1)
    #     ax_std.plot(periods_all, mean_std - ci_std, label='CI-', color='tab:green', linestyle='--', linewidth=1)
    #     ax_std.fill_between(periods_all, mean_std - ci_std, mean_std + ci_std, color='gray', alpha=0.2)
    
    # ax_std.set_title(f'{p_name}: Running Average (NO Warm-up Removal)', fontsize=14)
    # ax_std.set_xlabel('Month', fontsize=12)
    # ax_std.set_ylabel('Cost ($)', fontsize=12)
    # ax_std.grid(True, linestyle='--', alpha=0.6)
    
    # data_warmup = data_raw[:, warmup_period:] 
    # run_avg_wu = np.cumsum(data_warmup, axis=1) / np.arange(1, number_periods - warmup_period + 1)
    # mean_wu = np.mean(run_avg_wu, axis=0)
    # se_wu = st.sem(run_avg_wu, axis=0)
    # ci_wu = se_wu * st.t.ppf((1 + 0.95) / 2., K-1)
    
    # if p_name != "Policy 3":
    #     y_min = min(np.min(mean_std - ci_std), np.min(mean_wu - ci_wu)) - 20
    #     y_max = max(np.max(mean_std + ci_std), np.max(mean_wu + ci_wu)) + 20
    #     ax_std.set_ylim(y_min, y_max)
    #     ax_std.legend()
    # else:
    #     ax_std.set_ylim(550, 650)  
    
    # plt.tight_layout()
    # plt.savefig(f"Graph_{idx+1}B_{p_name.replace(' ', '_')}_NO_warmup.png")
    # plt.close() 

    # =========================================================
    # GRAPH C: WITH WARMUP REMOVAL (Multiple Replication CI)
    # =========================================================
    # results_wu.append({
    #     "Policy": p_name, 
    #     "Grand Mean Cost ($)": f"{mean_wu[-1]:.2f}", 
    #     "95% CI (+/-)": f"{ci_wu[-1]:.2f}",
    #     "95% CI Interval": f"[{mean_wu[-1] - ci_wu[-1]:.2f}, {mean_wu[-1] + ci_wu[-1]:.2f}]"
    # })

    # fig, ax_wu = plt.subplots(figsize=(8, 6))
    # periods_wu = np.arange(warmup_period + 1, number_periods + 1)
    
    # ax_wu.plot(periods_wu, mean_wu, label='Grand Mean', color='tab:blue', linewidth=2)
    
    # if p_name != "Policy 3":
    #     ax_wu.plot(periods_wu, mean_wu + ci_wu, label='CI+', color='tab:red', linestyle='--', linewidth=1)
    #     ax_wu.plot(periods_wu, mean_wu - ci_wu, label='CI-', color='tab:green', linestyle='--', linewidth=1)
    #     ax_wu.fill_between(periods_wu, mean_wu - ci_wu, mean_wu + ci_wu, color='gray', alpha=0.2)
    #     ax_wu.set_ylim(y_min, y_max)
    #     ax_wu.legend()
    # else:
    #     ax_wu.set_ylim(550, 650)
    
    # ax_wu.set_title(f'{p_name}: Running Average (WITH w={warmup_period} Removed)', fontsize=14)
    # ax_wu.set_xlabel('Month', fontsize=12)
    # ax_wu.set_ylabel('Cost ($)', fontsize=12)
    # ax_wu.grid(True, linestyle='--', alpha=0.6)
    # ax_wu.set_xlim(warmup_period, number_periods)
    # plt.tight_layout()
    # plt.savefig(f"Graph_{idx+1}C_{p_name.replace(' ', '_')}_WITH_warmup.png")
    # plt.close()

# --- 3. PRINT NUMERICAL RESULTS ---
df_std = pd.DataFrame(results_std)
print("--- MULTIPLE REPLICATION RESULTS (NO WARM-UP REMOVAL) ---")
print(df_std.to_string(index=False))
print("\n")

df_wu = pd.DataFrame(results_wu)
print(f"--- MULTIPLE REPLICATION RESULTS (WITH {warmup_period}-MONTH WARM-UP REMOVED) ---")
print(df_wu.to_string(index=False))