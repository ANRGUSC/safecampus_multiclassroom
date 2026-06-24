"""
Centralized PPO for Multi-Classroom Epidemic Control

A single central controller observes the global state (all classrooms' infected counts + risk)
and outputs actions for all classrooms simultaneously.

This version uses Beta policy (proper PPO):
- Stochastic policy with Beta distribution
- Actions naturally bounded to [0, 1] (no clipping!)
- Natural exploration through sampling
- Proper log probabilities for PPO

Author: SafeCampus Project
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import os
import json
import random
import time
import argparse

# pyrefly: ignore [missing-import]
from environment.multiclassroom import MultiClassroomEnv

# ============================================================
# 0. SETUP & CONFIGURATION
# ============================================================
GLOBAL_SEED = 42
OUTPUT_DIR = "centralized_ppo_results"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
HYPERPARAMS_FILE = os.path.join(OUTPUT_DIR, "optimized_hyperparams.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Hyperparameters
GAMMA_DISCOUNT = 0.99
GAE_LAMBDA = 0.95
K_EPOCHS = 20
EPS_CLIP = 0.2
MAX_WEEKS = 15
UPDATE_TIMESTEP = 2000
FULL_EPISODES = 3000
TUNE_EPISODES = 3000
LR_CANDIDATES = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01]
HIDDEN_DIM_CANDIDATES = [32, 64, 128]

# Omega (Preference weight)
OMEGA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Environment Settings
TOTAL_STUDENTS = 50
NUM_CLASSROOMS = 2
COOPERATIVE_REWARD = True
TUNE_SEED = 123
SHARED_FRACTION = 0.3

# Network Architecture
HIDDEN_DIM = 32
NUM_LAYERS = 2

# Evaluation
NUM_RUNS = 1

device = torch.device("cpu")
print(f"Training on device: {device}")

plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2.0,
})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize_global_state(observations, agent_ids, total_students):
    """Convert observations dict to normalized global state vector."""
    state = []
    for aid in agent_ids:
        obs = observations[aid]
        state.append(obs[0] / float(total_students))  # Normalize infected to [0, 1]
        state.append(obs[1])  # Risk already [0, 1]
    return np.array(state, dtype=np.float32)


# ============================================================
# 1. NETWORK ARCHITECTURES (Same as CTDE)
# ============================================================

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BetaActor(nn.Module):
    """
    Beta Actor for standard PPO.

    Outputs alpha and beta parameters for Beta distribution over actions.
    Beta distribution is naturally bounded to [0, 1] - no clipping needed!

    This is the same approach used in the successful ranking PPO.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super(BetaActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build feature network
        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())

        self.feature_net = nn.Sequential(*layers)

        # Alpha and Beta heads for Beta distribution
        self.alpha_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.beta_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, state):
        """
        Forward pass returning alpha and beta parameters.

        Returns:
            alpha: Alpha parameter (> 0)
            beta: Beta parameter (> 0)
        """
        features = self.feature_net(state)

        # Compute alpha and beta (must be > 0)
        # Use softplus + 1.0 to ensure alpha, beta > 1 (unimodal distribution)
        alpha = torch.nn.functional.softplus(self.alpha_head(features)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(features)) + 1.0

        return alpha, beta

    def act(self, state):
        """
        Sample action from Beta distribution.

        Returns:
            action: Sampled action in [0, 1] (naturally bounded!)
            log_prob: Log probability of the action
        """
        alpha, beta = self.forward(state)

        # Create Beta distribution (naturally bounded to [0, 1])
        dist = torch.distributions.Beta(alpha, beta)

        # Sample action (no clipping needed!)
        action = dist.sample()

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """
        Evaluate log probability and entropy for given state-action pairs.

        Args:
            state: State tensor
            action: Action tensor (already in [0, 1])

        Returns:
            log_prob: Log probabilities
            entropy: Distribution entropy
        """
        alpha, beta = self.forward(state)

        # Create Beta distribution
        dist = torch.distributions.Beta(alpha, beta)

        # Compute log probability (valid for any action in [0, 1])
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def get_deterministic_action(self, state):
        """Get deterministic action (mode of Beta distribution) for evaluation."""
        with torch.no_grad():
            alpha, beta = self.forward(state)

            # Mode of Beta distribution: (alpha - 1) / (alpha + beta - 2)
            # For alpha, beta > 1, mode is well-defined
            mode = (alpha - 1) / (alpha + beta - 2)
            mode = torch.clamp(mode, 0.0, 1.0)

            return mode

    # Alias for compatibility with evaluation scripts
    def get_deterministic_actions(self, state):
        return self.get_deterministic_action(state)


class Critic(nn.Module):
    """Value function critic."""

    def __init__(self, state_dim, hidden_dim=64, num_layers=2):
        super(Critic, self).__init__()

        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())
        layers.append(init_layer(nn.Linear(hidden_dim, 1), std=1.0))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# ============================================================
# 2. ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def __len__(self):
        return len(self.states)


# ============================================================
# 3. CENTRALIZED PPO AGENT
# ============================================================

class CentralizedPPO:
    """
    Centralized PPO that controls all agents with a single policy.

    Uses Beta policy for proper stochastic PPO (naturally bounded actions).
    """

    def __init__(self, global_state_dim, num_actions, lr,
                 hidden_dim=64, num_layers=2):

        self.global_state_dim = global_state_dim
        self.num_actions = num_actions
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gamma = GAMMA_DISCOUNT
        self.gae_lambda = GAE_LAMBDA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS

        # Actor and Critic networks (BetaActor for standard PPO)
        self.actor = BetaActor(global_state_dim, num_actions, hidden_dim, num_layers).to(device)
        self.critic = Critic(global_state_dim, hidden_dim, num_layers).to(device)

        # Old actor for PPO ratio computation
        self.actor_old = BetaActor(global_state_dim, num_actions, hidden_dim, num_layers).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

    def select_actions(self, global_state):
        """
        Select actions by sampling from Beta policy.

        Returns:
            actions: Actions in [0, 1] for each classroom
            log_prob: Log probability
            value: Value estimate for the state
        """
        state_tensor = torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            actions, log_prob = self.actor_old.act(state_tensor)
            value = self.critic(state_tensor)

        return actions.squeeze(0).cpu().numpy(), log_prob, value

    def update(self, buffer):
        """Update policy using PPO with GAE."""
        if len(buffer) == 0:
            return

        # Convert buffer data to tensors
        old_states = torch.stack(buffer.states).detach().to(device)
        old_actions = torch.stack(buffer.actions).detach().to(device)
        old_logprobs = torch.stack(buffer.logprobs).detach().to(device).squeeze()
        old_values = torch.stack(buffer.values).detach().to(device).squeeze()
        
        rewards = buffer.rewards
        is_terminals = buffer.is_terminals

        # Compute GAE advantages and returns
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                next_value = 0
                gae = 0
            else:
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = old_values[t + 1].item()

            delta = rewards[t] + self.gamma * next_value - old_values[t].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[t].item())

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.K_epochs):
            # Evaluate current policy
            logprobs, entropy = self.actor.evaluate(old_states, old_actions)
            logprobs = logprobs.squeeze()

            # Compute ratio (pi_new / pi_old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Actor loss with entropy bonus
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -0.01 * entropy.mean()  # Encourage exploration
            total_actor_loss = actor_loss + entropy_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # Critic loss
            state_values = self.critic(old_states).squeeze()
            critic_loss = self.MseLoss(state_values, returns)

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        # Sync old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        buffer.clear()

    def save(self, path):
        """Save model to file."""
        if not path.endswith('.pt'):
            path = path + '.pt'
            
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_old_state_dict': self.actor_old.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'global_state_dim': self.global_state_dim,
            'num_actions': self.num_actions,
            'lr': self.lr,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model from file."""
        if not path.endswith('.pt'):
            path = path + '.pt'
            
        checkpoint = torch.load(path, map_location=device)
        
        ppo = cls(
            global_state_dim=checkpoint['global_state_dim'],
            num_actions=checkpoint['num_actions'],
            lr=checkpoint['lr'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        )
        
        ppo.actor.load_state_dict(checkpoint['actor_state_dict'])
        ppo.actor_old.load_state_dict(checkpoint['actor_old_state_dict'])
        ppo.critic.load_state_dict(checkpoint['critic_state_dict'])
        ppo.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        ppo.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(f"Model loaded from {path}")
        return ppo

    # Compatibility property for evaluation scripts
    @property
    def policy(self):
        """Compatibility alias - returns actor."""
        return self.actor


# ============================================================
# 4. TRAINING FUNCTIONS
# ============================================================

def run_centralized_training(omega, seed, lr, episodes, num_classrooms=NUM_CLASSROOMS, hidden_dim=HIDDEN_DIM, shared_fraction=SHARED_FRACTION):
    """Run centralized PPO training."""
    set_seed(seed)

    env = MultiClassroomEnv(
        num_classrooms=num_classrooms,
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        gamma=omega,
        continuous_action=True,
        cooperative_reward=COOPERATIVE_REWARD,
        shared_fraction=shared_fraction
    )

    agent_ids = sorted(env.agents)
    global_state_dim = 2 * num_classrooms  # (infected, risk) per classroom
    num_actions = num_classrooms

    ppo = CentralizedPPO(
        global_state_dim=global_state_dim,
        num_actions=num_actions,
        lr=lr,
        hidden_dim=hidden_dim,
        num_layers=NUM_LAYERS
    )

    buffer = RolloutBuffer()
    reward_history = []
    time_step = 0

    for ep in range(episodes):
        obs = env.reset()
        global_state = normalize_global_state(obs, agent_ids, TOTAL_STUDENTS)
        ep_reward = 0
        done = False

        while not done:
            time_step += 1

            actions_normalized, log_prob, value = ppo.select_actions(global_state)

            actions_env = {
                aid: np.array([actions_normalized[i] * TOTAL_STUDENTS])
                for i, aid in enumerate(agent_ids)
            }

            next_obs, rewards, dones, _ = env.step(actions_env)

            joint_reward = sum(rewards.values()) / len(rewards)

            # Store in buffer
            buffer.states.append(torch.tensor(global_state, dtype=torch.float32))
            buffer.actions.append(torch.tensor(actions_normalized, dtype=torch.float32))
            buffer.logprobs.append(log_prob)
            buffer.values.append(value.squeeze())
            buffer.rewards.append(joint_reward)
            buffer.is_terminals.append(any(dones.values()))

            global_state = normalize_global_state(next_obs, agent_ids, TOTAL_STUDENTS)
            ep_reward += joint_reward

            if time_step % UPDATE_TIMESTEP == 0:
                ppo.update(buffer)

            done = any(dones.values())

        reward_history.append(ep_reward)

    return ppo, reward_history


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================

def select_best_hyperparams(omega_results):
    """Select best hyperparameters based on reward."""
    sorted_results = sorted(omega_results, key=lambda r: -r['avg_eval_reward'])

    best = sorted_results[0]
    return (best['lr'], best['hidden_dim']), best, {}


def grid_search_tuning(num_classrooms=NUM_CLASSROOMS, shared_fraction=SHARED_FRACTION):
    """Grid search for best learning rate and hidden dimension per omega."""
    optimized_hyperparams = {}

    print(f"\n--- Starting Grid Search Tuning (Beta Policy) ---")
    print(f"LR Candidates: {LR_CANDIDATES}")
    print(f"Hidden Dim Candidates: {HIDDEN_DIM_CANDIDATES}")
    print(f"Number of Classrooms: {num_classrooms}")

    for omega in OMEGA_VALUES:
        print(f"\n{'=' * 60}")
        print(f"*** Tuning Omega = {omega} ***")
        print(f"{'=' * 60}")

        omega_results = []

        for lr in LR_CANDIDATES:
            for hidden_dim in HIDDEN_DIM_CANDIDATES:
                print(f"\n  Testing LR={lr}, Hidden Dim={hidden_dim}...")

                ppo, history = run_centralized_training(omega, TUNE_SEED, lr, TUNE_EPISODES, num_classrooms, hidden_dim, shared_fraction)

                # Evaluation
                env = MultiClassroomEnv(
                    num_classrooms=num_classrooms,
                    total_students=TOTAL_STUDENTS,
                    max_weeks=MAX_WEEKS,
                    gamma=omega,
                    continuous_action=True,
                    cooperative_reward=COOPERATIVE_REWARD,
                    eval_mode=True,
                    shared_fraction=shared_fraction
                )
                agent_ids = sorted(env.agents)
                obs = env.reset()
                global_state = normalize_global_state(obs, agent_ids, TOTAL_STUDENTS)
                done = False
                avg_eval_reward = 0

                while not done:
                    state_tensor = torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        actions_normalized = ppo.actor.get_deterministic_action(state_tensor).cpu().numpy().flatten()
                    actions_env = {aid: np.array([actions_normalized[i] * TOTAL_STUDENTS]) for i, aid in enumerate(agent_ids)}
                    next_obs, rewards, dones, _ = env.step(actions_env)
                    avg_eval_reward += sum(rewards.values()) / len(rewards)
                    global_state = normalize_global_state(next_obs, agent_ids, TOTAL_STUDENTS)
                    done = any(dones.values())

                result = {
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'avg_eval_reward': avg_eval_reward,
                }
                omega_results.append(result)

                print(f"    Reward: {avg_eval_reward:.2f}")

        best_hyperparams, best_metrics, _ = select_best_hyperparams(omega_results)
        best_lr, best_hidden_dim = best_hyperparams
        optimized_hyperparams[str(omega)] = {'lr': best_lr, 'hidden_dim': best_hidden_dim}
        print(f"\n  --> Best LR: {best_lr}, Best Hidden Dim: {best_hidden_dim}")

    with open(HYPERPARAMS_FILE, 'w') as f:
        json.dump(optimized_hyperparams, f, indent=4)

    return {float(k): v for k, v in optimized_hyperparams.items()}


def load_hyperparams():
    """Load optimized hyperparameters from file."""
    if os.path.exists(HYPERPARAMS_FILE):
        with open(HYPERPARAMS_FILE, 'r') as f:
            return {float(k): v for k, v in json.load(f).items()}
    return {omega: {'lr': 0.001, 'hidden_dim': 32} for omega in OMEGA_VALUES}


# ============================================================
# 6. FULL TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate_optimal(optimized_hyperparams, shared_fraction=SHARED_FRACTION, num_classrooms=NUM_CLASSROOMS):
    """Run full training with optimized hyperparameters and save models."""
    print(f"\n--- Starting Full Training (Beta Policy) ---")
    print(f"Number of Classrooms: {num_classrooms}")

    all_rewards = {}
    trained_agents = {}

    for omega in OMEGA_VALUES:
        hyperparams = optimized_hyperparams.get(omega, {'lr': 0.001, 'hidden_dim': 32})
        lr = hyperparams['lr']
        hidden_dim = hyperparams['hidden_dim']
        print(f"\n{'=' * 60}")
        print(f"*** Training Omega = {omega}, LR = {lr}, Hidden Dim = {hidden_dim} ***")
        print(f"{'=' * 60}")

        for run in range(NUM_RUNS):
            seed = TUNE_SEED
            print(f"\n  Run {run + 1}/{NUM_RUNS} (seed={seed})")

            ppo, history = run_centralized_training(omega, seed, lr, FULL_EPISODES, num_classrooms, hidden_dim, shared_fraction)

            # Save model
            model_path = os.path.join(MODEL_DIR, f"centralized_omega_{omega}_sf_{shared_fraction}_k_{num_classrooms}_hd_{hidden_dim}_run_{run}")
            ppo.save(model_path)

            if run == 0:
                all_rewards[omega] = np.array([history])
                trained_agents[omega] = ppo

        print(f"  Final Reward: {np.mean(history[-50:]):.2f}")

    # Plot results
    plot_combined_rewards(all_rewards)

    print(f"\nTraining complete. Models saved to {MODEL_DIR}")

    return trained_agents


# ============================================================
# 7. PLOTTING FUNCTIONS
# ============================================================

def plot_combined_rewards(all_rewards):
    """Plot training rewards."""
    print("\n--- Plotting Rewards ---")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(OMEGA_VALUES)))

    for idx, omega in enumerate(OMEGA_VALUES):
        data = all_rewards[omega]
        mean_rewards = np.mean(data, axis=0)

        window = 100
        kernel = np.ones(window) / window
        smoothed = np.convolve(mean_rewards, kernel, mode='valid')

        plt.plot(smoothed, label=f'ω={omega}', color=colors[idx], linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Centralized PPO Training Rewards (Beta Policy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_rewards.png"), dpi=300)
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================

def main(mode='tune_and_train', num_classrooms = NUM_CLASSROOMS, shared_fraction=SHARED_FRACTION, limit_omega=False):
    """
    Main function.

    Modes:
    - 'tune': Grid search for best hyperparameters (LR and Hidden Dim)
    - 'train': Train with saved hyperparameters
    - 'tune_and_train': Both
    """

    if limit_omega:
        global OMEGA_VALUES
        OMEGA_VALUES = [0.2, 0.4]


    print(f"\n{'='*60}")
    print(f"Centralized PPO Training (Beta Policy)")
    print(f"Number of Classrooms: {num_classrooms}")
    print(f"{'='*60}")

    if mode in ['tune', 'tune_and_train']:
        optimized_hyperparams = grid_search_tuning(num_classrooms=num_classrooms, shared_fraction=shared_fraction)
    else:
        optimized_hyperparams = load_hyperparams()

    if mode in ['train', 'tune_and_train']:
        train_and_evaluate_optimal(optimized_hyperparams, num_classrooms=num_classrooms, shared_fraction=shared_fraction)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_classrooms",
        type = int,
        default = NUM_CLASSROOMS,
        help =  "Number of classrooms used in experiment (default:2)"
    )
    
    parser.add_argument(
        "--shared_fraction",
        type = float,
        default = SHARED_FRACTION,
        help = "Controls how connected the classrooms are lower is isolated, higher has a risk of spillover (default:0.3)"
    )

    parser.add_argument(
        "--limit_omega",
        action="store_true",
        help = "If flag included restricts omega values to [0.2, 0.4] for faster training (default:False)"
    )

    args = parser.parse_args()

    main(mode='tune_and_train', num_classrooms = args.num_classrooms, shared_fraction=args.shared_fraction, limit_omega=args.limit_omega)