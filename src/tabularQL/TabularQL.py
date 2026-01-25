"""
Tabular Q-learning on the Blackjack-v1 environment (gymnasium)
"""

import argparse
import collections
import math
import pickle
import random
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# try gymnasium first; older systems may have gym
try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

State = Tuple[int, int, bool]  # (player_sum, dealer_card, usable_ace)
Action = int  # 0 = stick, 1 = hit

#return next state from environment when applying an action
def make_state_key(obs) -> State:
    """
    Gymnasium Blackjack returns observations as (player_sum, dealer_card, usable_ace)
    We use the raw tuple as a hashable state key.
    """
    return (int(obs[0]), int(obs[1]), bool(obs[2]))

class TabularQAgent:
    def __init__(self,
                 actions: List[Action],
                 alpha: float = 0.1,
                 gamma: float = 1.0,
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 1.0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: dict[state] -> np.array([Q(s,0), Q(s,1), ...])
        self.Q: Dict[State, np.ndarray] = {}
    
    #get Q(s,...) = array of actions for given state s and relative values
    def get_qs(self, s: State) -> np.ndarray:
        if s not in self.Q:
            # initialize optimistic (0.0) values for known small space
            self.Q[s] = np.zeros(len(self.actions), dtype=float)
        return self.Q[s]
    
    # epsilon-greedy strategy
    def choose_action(self, s: State) -> Action:
        #random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        #best action
        qs = self.get_qs(s)
        # break ties randomly in case multiple actions yeld max return
        maxv = np.max(qs)
        candidates = [a for a, q in enumerate(qs) if q == maxv]
        return random.choice(candidates)
    
    # update rule: q(s,a) = Q(s,a) + alpha * (r + gamma * max_a'Q(s',a'))
    def update(self, s: State, a: Action, r: float, s2: State, done: bool):
        q = self.get_qs(s)[a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.get_qs(s2))
        td = target - q
        self.get_qs(s)[a] += self.alpha * td
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    """ #save Q-Table in file
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)
        #print(f"Saved Q-table to {path} (states: {len(self.Q)})")
    #load Q-Table from file
    def load(self, path: str):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
        #print(f"Loaded Q-table from {path} (states: {len(self.Q)})") """


def train(env,
          agent: TabularQAgent,
          episodes: int = 200_000,
          stats_window: int = 10_000,
          render_every: int = 0):
    """
    Train the Q-learning agent. Returns training statistics.

    stats collected per episode:
      - reward (float)
      - result (win/draw/loss) based on reward: +1 -> win, 0 -> draw, -1 -> loss
    """
    rewards = np.zeros(episodes, dtype=float)
    results = np.zeros(episodes, dtype=int)  # 1 win, 0 draw, -1 loss
    epsilons = np.zeros(episodes, dtype=float)
    
    wins = 0
    draws = 0
    losses = 0

    pbar = trange(episodes, desc="Training", unit="ep")
    for ep in pbar:
        obs, _ = env.reset()
        s = make_state_key(obs)
        done = False
        total_reward = 0.0

        while not done:
            a = agent.choose_action(s)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s2 = make_state_key(obs2)
            agent.update(s, a, r, s2, done)
            s = s2
            total_reward += r

        rewards[ep] = total_reward
        if total_reward > 0:
            results[ep] = 1
            wins += 1
        elif total_reward == 0:
            results[ep] = 0
            draws += 1
        else:
            results[ep] = -1
            losses += 1

        epsilons[ep] = agent.epsilon
        agent.decay_epsilon()

        # progress bar stats
        if (ep + 1) % max(1, episodes // 100) == 0:
            recent_mean = rewards[max(0, ep - stats_window + 1):(ep + 1)].mean()
            pbar.set_postfix({
                "ep": ep + 1,
                "recent_mean_reward": f"{recent_mean:.4f}",
                "eps": f"{agent.epsilon:.5f}"
            })

        if render_every and (ep + 1) % render_every == 0:
            # optionally render episode (slow)
            env.render()
    
    stats = {
        "rewards": rewards,
        "results": results,
        "epsilons": epsilons,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }
    return agent, stats

def evaluate_agent(env, agent, episodes: int = 200_000):
    """
    Evaluate the agent with epsilon=0 (fully greedy policy).
    Returns win/draw/loss counts and average reward.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # turn off exploration

    wins = draws = losses = 0
    total_reward = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        s = make_state_key(obs)
        done = False

        while not done:
            # greedy action: pick max Q
            qs = agent.get_qs(s)
            a = np.argmax(qs)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = make_state_key(obs2)
            total_reward += r

        if r > 0:
            wins += 1
        elif r == 0:
            draws += 1
        else:
            losses += 1

    # restore original exploration rate
    agent.epsilon = original_epsilon

    stats = {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "loss_rate": losses / episodes,
        "avg_reward": total_reward / episodes
    }
    return stats

def plot_custom_stats(stats, alpha, epsilon, epsilon_min, epsilon_decay, gamma, episodes):
    rewards = stats["rewards"]
    results = stats["results"]   # 1=win, 0=draw, -1=loss
    epsilons = stats["epsilons"]
    wins = stats["wins"]
    draws = stats["draws"]
    losses = stats["losses"]

    # Success metric: win = 1, everything else = 0
    successes = (results == 1).astype(int)

    # Trend lines via moving average
    window = max(500, episodes // 200)  # adaptive smoothing window
    def moving_average(x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w)/w, mode='same')

    rewards_ma = moving_average(rewards, window)
    successes_ma = moving_average(successes, window)

    # ---- Plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"alpha={alpha}, epsilon:{epsilon} - {epsilon_min}, epsilon_decay={epsilon_decay}, gamma={gamma}, episodes={episodes}",
        fontsize=14
    )

    # (1) Epsilon decay
    ax = axes[0, 0]
    ax.plot(epsilons, label="epsilon")
    ax.set_title("Epsilon decay over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("epsilon")
    ax.grid(True)

    # (2) Reward per episode + trend
    ax = axes[0, 1]
    ax.plot(rewards, alpha=0.4, label="reward")
    ax.plot(rewards_ma, color='red', linewidth=2, label=f"moving avg (w={window})")
    ax.set_title("Reward per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)

    # (3) Success per episode + trend
    ax = axes[1, 0]
    ax.plot(successes, alpha=0.3, label="success (1 = win)")
    ax.plot(successes_ma, color='green', linewidth=2, label=f"moving avg (w={window})")
    ax.set_title("Win successes per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win (1=yes, 0=no)")
    ax.legend()
    ax.grid(True)

    # (4) Pie chart of win/draw/loss percentages
    ax = axes[1, 1]
    labels = ["Wins", "Draws", "Losses"]
    counts = [wins, draws, losses]
    colors = ["#4CAF50", "#2196F3", "#F44336"]  # optional aesthetic colors
    ax.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    ax.set_title("Outcome percentages: wins "+str(wins)+", losses "+str(losses)+", draws "+str(draws))

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{episodes}ep_{alpha}a_{gamma}g_{epsilon}e_{epsilon_decay}edec__tabularQ.png', dpi=300, bbox_inches='tight')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.999995)
    p.add_argument("--stats-window", type=int, default=10000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", type=str, default="q_table.pkl")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Gymnasium: Blackjack-v1
    env = gym.make("Blackjack-v1", sab=True) if "gymnasium" in gym.__name__ else gym.make("Blackjack-v1")
    # sab=True ensures "natural" blackjack payout handling consistent across versions
    actions = [0, 1]  # stick, hit

    agent = TabularQAgent(actions=actions,
                          alpha=args.alpha,
                          gamma=args.gamma,
                          epsilon=args.epsilon,
                          epsilon_min=args.epsilon_min,
                          epsilon_decay=args.epsilon_decay)

    print(f"Starting training: episodes={args.episodes}, alpha={args.alpha}, gamma={args.gamma}, "
          f"epsilon={args.epsilon}, eps_min={args.epsilon_min}, eps_decay={args.epsilon_decay}")
    agent, stats = train(env, agent, episodes=args.episodes, stats_window=args.stats_window)
    print(f"Training finished: wins={stats['wins']}, draws={stats['draws']}, losses={stats['losses']}")

    # save Q-table
    #agent.save(args.save)

    plot_custom_stats(stats, alpha=args.alpha, epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay, gamma=args.gamma, episodes=args.episodes)

    stats = evaluate_agent(env, agent, episodes=args.episodes)
    print(f"Evaluation: \n   win_rate: {stats["win_rate"]}\n   loss_rate: {stats["loss_rate"]}\n   draw_rate: {stats["draw_rate"]}\n   avg_reward: {stats["avg_reward"]}")

    env.close()

if __name__ == "__main__":
    main()
