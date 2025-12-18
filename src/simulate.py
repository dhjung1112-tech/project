import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Config: states, actions, transition matrix, rewards
# -----------------------------
STATES = ["LOW", "MID", "HIGH"]  # energy level
ACTIONS = ["STUDY", "EXERCISE", "REST", "MOVIE"]

# Transition matrix P[i, j] = Pr(next_state=j | current_state=i)
P = np.array([
    [0.55, 0.35, 0.10],  # from LOW
    [0.20, 0.55, 0.25],  # from MID
    [0.10, 0.35, 0.55],  # from HIGH
], dtype=float)

# Reward table r[state, action]
# rows: LOW/MID/HIGH, cols: STUDY/EXERCISE/REST/MOVIE
R = np.array([
    [2,  1,  6, 4],  # LOW
    [5,  4,  4, 5],  # MID
    [8,  7,  2, 4],  # HIGH
], dtype=float)


# -----------------------------
# 2) Utilities
# -----------------------------
def sample_next_state(curr_state_idx: int, rng: np.random.Generator) -> int:
    probs = P[curr_state_idx]
    return int(rng.choice(len(STATES), p=probs))

def reward(state_idx: int, action_idx: int) -> float:
    return float(R[state_idx, action_idx])

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# 3) Policies
# -----------------------------
def policy_random(state_idx: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(ACTIONS)))

def policy_greedy(state_idx: int, rng: np.random.Generator) -> int:
    # choose action with max immediate reward at current state
    return int(np.argmax(R[state_idx]))

def policy_heuristic(state_idx: int, rng: np.random.Generator) -> int:
    # simple rules:
    # - if LOW: REST is preferred
    # - if HIGH: STUDY is preferred
    # - if MID: EXERCISE slightly preferred, otherwise MOVIE
    if state_idx == 0:  # LOW
        return ACTIONS.index("REST")
    if state_idx == 2:  # HIGH
        return ACTIONS.index("STUDY")
    # MID
    return ACTIONS.index("EXERCISE")


POLICIES = {
    "random_baseline": policy_random,
    "greedy": policy_greedy,
    "heuristic": policy_heuristic,
}


# -----------------------------
# 4) Simulation
# -----------------------------
def run_episode(policy_fn, horizon: int, start_state_idx: int, rng: np.random.Generator):
    state = start_state_idx
    total = 0.0
    state_counts = np.zeros(len(STATES), dtype=int)

    for _ in range(horizon):
        state_counts[state] += 1
        action = policy_fn(state, rng)
        total += reward(state, action)
        state = sample_next_state(state, rng)

    return total, state_counts

def monte_carlo(policy_name: str, n_episodes: int = 3000, horizon: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    policy_fn = POLICIES[policy_name]

    totals = []
    state_counts_sum = np.zeros(len(STATES), dtype=int)

    # randomize start states for fairness
    for _ in range(n_episodes):
        start_state = int(rng.integers(0, len(STATES)))
        total, state_counts = run_episode(policy_fn, horizon, start_state, rng)
        totals.append(total)
        state_counts_sum += state_counts

    return {
        "policy": policy_name,
        "avg_total_reward": float(np.mean(totals)),
        "std_total_reward": float(np.std(totals, ddof=1)),
        "state_dist": state_counts_sum / state_counts_sum.sum()
    }


# -----------------------------
# 5) Main
# -----------------------------
def main():
    n_episodes = 5000
    horizon = 30
    seed = 42

    results = [monte_carlo(name, n_episodes=n_episodes, horizon=horizon, seed=seed) for name in POLICIES]
    df = pd.DataFrame(results).sort_values("avg_total_reward", ascending=False)

    print(f"Policy Comparison (n_episodes={n_episodes}, horizon={horizon})")
    for _, row in df.iterrows():
        print(f"- {row['policy']:<15} avg={row['avg_total_reward']:.2f}  std={row['std_total_reward']:.2f}")
    print(f"Best policy: {df.iloc[0]['policy']}")

    # Save plot
    ensure_dir("outputs/figures")

    plt.figure()
    plt.bar(df["policy"], df["avg_total_reward"])
    plt.title("Average Total Reward by Policy")
    plt.ylabel("Average total reward")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_path = "outputs/figures/policy_comparison.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")

    # Also print state distribution for best policy
    best = df.iloc[0]
    best_dist = best["state_dist"]
    print("\nState distribution (best policy):")
    for s, p in zip(STATES, best_dist):
        print(f"- {s}: {p:.3f}")


if __name__ == "__main__":
    main()
