"""
DQN on the Blackjack-v1 environment (gymnasium)
"""

import argparse
import collections
import math
import random
from typing import Deque, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
# PyTorch for neural nets
import torch
import torch.nn as nn
import torch.optim as optim

# try gymnasium first; older systems may have gym
try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

# ---------- utils for states ----------
def make_state_key(obs) -> Tuple[int, int, bool]:
    return (int(obs[0]), int(obs[1]), bool(obs[2]))

def state_to_tensor(obs, device):
    """
    Convert state tuple (player_sum, dealer_card, usable_ace)
    to a scaled float tensor for NN input.
    Scaling choices:
      - player_sum roughly 4..21 -> divide by 32
      - dealer_card 1..10 -> divide by 11
      - usable_ace -> 0/1
    Returns tensor shape (1, 3) for single state or (N,3) for batched numpy arrays.
    """
    arr = np.array([obs[0] / 32.0, obs[1] / 11.0, 1.0 if obs[2] else 0.0], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # shape (1,3)

# ---------- Replay buffer definition ----------
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.vstack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)

# ---------- DQN network ----------
# N.N. definition
# 2 hidden layers, 3 inputs: state, 2 outputs: actions (hit/stick), ReLU activ. units
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = [32, 32]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Agent using target DQN ----------
##############################################
"""class DQNAgent:
    def __init__(self,
                 actions: List[int],
                 device: torch.device,
                 gamma: float = 1.0,
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.999995,
                 lr: float = 1e-3,
                 replay_size: int = 10000,
                 batch_size: int = 32,
                 target_update: int = 1000):
        self.actions = actions
        self.n_actions = len(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = device

        # small network: input 3 features, output n_actions
        self.policy_net = DQN(input_dim=3, output_dim=self.n_actions).to(device)
        self.target_net = DQN(input_dim=3, output_dim=self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss (robust)

        # replay
        self.replay = ReplayBuffer(replay_size)
        self.batch_size = batch_size

        # bookkeeping for target network updates
        self.steps_done = 0
        self.target_update = target_update

    def choose_action(self, obs) -> int:
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # greedy from policy_net
        with torch.no_grad():
            s_t = state_to_tensor(obs, self.device)  # (1,3)
            qvals = self.policy_net(s_t)  # (1, n_actions)
            action = int(torch.argmax(qvals, dim=1).item())
            return action

    def store_transition(self, s, a, r, s2, done):
        # store numpy-form states: shape (1,3) arrays
        s_arr = np.array([s[0] / 32.0, s[1] / 11.0, 1.0 if s[2] else 0.0], dtype=np.float32)
        s2_arr = np.array([s2[0] / 32.0, s2[1] / 11.0, 1.0 if s2[2] else 0.0], dtype=np.float32)
        self.replay.push(s_arr, a, r, s2_arr, done)

    def update_model(self):
        # run one optimization step if enough transitions
        if len(self.replay) < self.batch_size:
            return

        states_np, actions_np, rewards_np, next_states_np, dones_np = self.replay.sample(self.batch_size)

        states = torch.from_numpy(states_np).to(self.device)           # (B,3)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)  # (B,1)
        rewards = torch.from_numpy(rewards_np).unsqueeze(1).to(self.device)  # (B,1)
        next_states = torch.from_numpy(next_states_np).to(self.device) # (B,3)
        dones = torch.from_numpy(dones_np).unsqueeze(1).to(self.device)      # (B,1) 0/1

        # current Q values
        q_values = self.policy_net(states).gather(1, actions)  # (B,1)

        # compute next state's max Q from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]  # (B,1)
            target_q = rewards + (1.0 - dones.float()) * (self.gamma * next_q_values)

        loss = self.loss_fn(q_values, target_q)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to keep stable
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # maybe update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
"""
# ---------- Agent using single DQN ----------
##############################################
class SingleNetworkDQNAgent:
    def __init__(self,
                 actions: List[int],
                 device: torch.device,
                 gamma: float = 1.0,
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.999995,
                 lr: float = 1e-3,
                 replay_size: int = 10000,
                 batch_size: int = 32):
        
        self.actions = actions
        self.n_actions = len(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device

        # Only ONE neural network
        self.q_net = DQN(input_dim=3, output_dim=self.n_actions).to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(replay_size)
        self.batch_size = batch_size

    # action selection with e-greedy strategy
    def choose_action(self, obs) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        with torch.no_grad():
            s_t = state_to_tensor(obs, self.device)
            a = int(torch.argmax(self.q_net(s_t), dim=1))
            return a

    #store (state, action, reward, next_state) in replay buffer
    def store_transition(self, s, a, r, s2, done):
        s_arr = np.array([s[0]/32.0, s[1]/11.0, float(s[2])], dtype=np.float32)
        s2_arr = np.array([s2[0]/32.0, s2[1]/11.0, float(s2[2])], dtype=np.float32)
        self.replay.push(s_arr, a, r, s2_arr, done)

    def update_model(self):
        if len(self.replay) < self.batch_size:
            return

        states_np, actions_np, rewards_np, next_states_np, dones_np = self.replay.sample(self.batch_size)
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards_np).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).unsqueeze(1).float().to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # Q-learning target computed from the SAME network
        with torch.no_grad():
            max_next_q = self.q_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ---------- Training loop ----------
def train(env,
          agent: SingleNetworkDQNAgent,
          episodes: int = 200_000,
          stats_window: int = 10_000,
          render_every: int = 0):
    rewards = np.zeros(episodes, dtype=float)
    results = np.zeros(episodes, dtype=int)  # 1 win, 0 draw, -1 loss
    epsilons = np.zeros(episodes, dtype=float)

    wins = draws = losses = 0

    pbar = trange(episodes, desc="Training", unit="ep")
    for ep in pbar:
        obs, _ = env.reset()
        s = make_state_key(obs)
        done = False
        total_reward = 0.0

        while not done:
            a = agent.choose_action(obs)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s2 = make_state_key(obs2)

            # store transition and (maybe) update
            agent.store_transition(s, a, r, s2, done)
            agent.update_model()

            obs = obs2
            s = s2
            total_reward += r

        # end episode bookkeeping
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

# ---------- Evaluation ----------
def evaluate_agent(env, agent: SingleNetworkDQNAgent, episodes: int = 200_000):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # NO GREEDY

    wins = draws = losses = 0
    total_reward = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        last_r = 0.0
        while not done:
            # obs carrys the state information
            a = agent.choose_action(obs)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            obs = obs2
            
            last_r = r
            total_reward += r

        if last_r > 0:
            wins += 1
        elif last_r == 0:
            draws += 1
        else:
            losses += 1

    agent.epsilon = original_epsilon

    stats = {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "loss_rate": losses / episodes,
        "avg_reward": total_reward / episodes
    }
    return stats

# ---------- plotting kept same ----------
def plot_custom_stats(stats, alpha, epsilon, epsilon_min, epsilon_decay, gamma, episodes, replay_size, batch_size):
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
        f"alpha={alpha}, epsilon:{epsilon} - {epsilon_min}, epsilon_decay={epsilon_decay}, gamma={gamma}, episodes={episodes}, buffer_size={replay_size}, minibatch_size={batch_size}",
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
    plt.savefig(f'{episodes}ep_{alpha}a_{gamma}g_{epsilon}e_{epsilon_decay}edec_{replay_size}buff_{batch_size}minib__DQN.png', dpi=300, bbox_inches='tight')

# ---------- argument parsing ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200_000)
    # "alpha" doesn't exist for DQN but keep the CLI param to keep compatibility with your plotting header
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.999995)
    p.add_argument("--stats-window", type=int, default=10000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--replay-size", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    #p.add_argument("--lr", type=float, default=1e-3)
    #p.add_argument("--target-update", type=int, default=1000)
    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gymnasium: Blackjack-v1
    env = gym.make("Blackjack-v1", sab=True) if "gymnasium" in gym.__name__ else gym.make("Blackjack-v1")
    actions = [0, 1]  # stick, hit

    # target_update=args.target_update  
    # param for standard DQN (2 NN)
    agent = SingleNetworkDQNAgent(actions=actions,
                     device=device,
                     gamma=args.gamma,
                     epsilon=args.epsilon,
                     epsilon_min=args.epsilon_min,
                     epsilon_decay=args.epsilon_decay,
                     lr=args.alpha,
                     replay_size=args.replay_size,
                     batch_size=args.batch_size
                     )

    print(f"Starting DQN training: episodes={args.episodes}, gamma={args.gamma}, "
          f"epsilon={args.epsilon}, eps_min={args.epsilon_min}, eps_decay={args.epsilon_decay}, "
          f"replay_size={args.replay_size}, batch_size={args.batch_size}, lr={args.alpha}")

    agent, stats = train(env, agent, episodes=args.episodes, stats_window=args.stats_window)
    print(f"Training finished: wins={stats['wins']}, draws={stats['draws']}, losses={stats['losses']}")

    plot_custom_stats(stats, alpha=args.alpha,
         epsilon=args.epsilon,
         epsilon_min=args.epsilon_min, 
         epsilon_decay=args.epsilon_decay,
         gamma=args.gamma,
         episodes=args.episodes,
         replay_size=args.replay_size,
         batch_size=args.batch_size
         )

    env.close()

if __name__ == "__main__":
    main()
