"""
MAPPO CTDE (Centralized Training, Decentralized Execution) for Multi-Classroom Epidemic Control

This implementation supports multiple policy types for continuous action spaces:

Policy options:
- Gaussian: Standard MAPPO with Gaussian distribution - most common in literature
- Beta: Beta distribution naturally bounded to [0, 1] - no clipping needed
- Tanh: Deterministic policy with tanh squashing and exploration noise

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
from torch.distributions import Beta
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import os
import json
import random
import time
# pyrefly: ignore [missing-import]
from matplotlib.colors import LinearSegmentedColormap
import colorsys

# pyrefly: ignore [missing-import]
from environment.multiclassroom import MultiClassroomEnv

# ============================================================
# 0. SETUP & CONFIGURATION
# ============================================================
GLOBAL_SEED = 42
OUTPUT_DIR = "mappo_results"
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

# Omega (Preference) - The weight 'gamma' in the environment
OMEGA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Environment Settings
TOTAL_STUDENTS = 50
NUM_CLASSROOMS = 2
COOPERATIVE_REWARD = True
TUNE_SEED = 123
# TODO (Student): You will need to add a default SHARED_FRACTION = 0.3 here
# so you have a default value to use if the command line argument isn't provided.

# Policy Grid Config (for policy heatmap plots)
POLICY_GRID_POINTS = 20

# Network Architecture Config
ACTOR_HIDDEN_DIM = 64
CRITIC_HIDDEN_DIM = 64
ACTOR_NUM_LAYERS = 2
CRITIC_NUM_LAYERS = 2

# Multi-run config
NUM_RUNS = 1

device = torch.device("cpu")
print(f"Training on device: {device}")

# Matplotlib formatting
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2.0,
    'figure.titlesize': 14
})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize_state(obs, total_students):
    """Normalize state: infected count to [0,1], risk already in [0,1]."""
    return np.array([obs[0] / float(total_students), obs[1]], dtype=np.float32)


# ============================================================
# 1. NETWORK ARCHITECTURES
# ============================================================

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BetaActor(nn.Module):
    """
    Beta Distribution Actor for bounded continuous action spaces.
    
    The Beta distribution is defined on [0, 1] with parameters alpha > 0 and beta > 0.
    - alpha = beta = 1: Uniform distribution
    - alpha > beta: Skewed towards 1
    - alpha < beta: Skewed towards 0
    - alpha = beta > 1: Peaked at 0.5
    
    We parameterize alpha and beta using softplus to ensure they're positive.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super(BetaActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared feature extractor
        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())
        
        self.feature_net = nn.Sequential(*layers)
        
        # Separate heads for alpha and beta parameters
        # Initialize to produce alpha=beta≈2 (slightly peaked at 0.5)
        self.alpha_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.beta_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)
        
        # Softplus for positive outputs, with minimum value for numerical stability
        self.softplus = nn.Softplus()
        self.min_param = 1.0  # Minimum alpha/beta value

    def forward(self, state):
        """
        Forward pass returning alpha and beta parameters.
        
        Returns:
            alpha: Beta distribution alpha parameter (> 1)
            beta: Beta distribution beta parameter (> 1)
        """
        features = self.feature_net(state)
        
        # Apply softplus and add minimum to ensure alpha, beta > 1
        alpha = self.softplus(self.alpha_head(features)) + self.min_param
        beta = self.softplus(self.beta_head(features)) + self.min_param
        
        return alpha, beta

    def act(self, state):
        """
        Sample action for execution.
        
        Returns:
            action: Sampled action in [0, 1]
            log_prob: Log probability of the action
        """
        alpha, beta = self.forward(state)
        
        dist = Beta(alpha, beta)
        action = dist.sample()
        
        # Clamp to avoid numerical issues at boundaries
        action = torch.clamp(action, 1e-6, 1 - 1e-6)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """
        Evaluate log probability and entropy for given state-action pairs.
        
        Args:
            state: Batch of states
            action: Batch of actions in [0, 1]
            
        Returns:
            log_prob: Log probabilities
            entropy: Distribution entropy
        """
        alpha, beta = self.forward(state)
        
        dist = Beta(alpha, beta)
        
        # Clamp actions for numerical stability
        action_clamped = torch.clamp(action, 1e-6, 1 - 1e-6)
        
        log_prob = dist.log_prob(action_clamped).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy

    def get_deterministic_action(self, state):
        """
        Get deterministic action (mode of Beta distribution) for evaluation.
        
        Mode of Beta(alpha, beta) = (alpha - 1) / (alpha + beta - 2) when alpha, beta > 1
        """
        with torch.no_grad():
            alpha, beta = self.forward(state)
            
            # Mode of Beta distribution (valid when alpha, beta > 1)
            mode = (alpha - 1) / (alpha + beta - 2)
            
            # Clamp to valid range
            mode = torch.clamp(mode, 0.0, 1.0)
            
            return mode


class TanhDeterministicActor(nn.Module):
    """
    Deterministic Actor with Tanh output squashing.

    This is a simpler alternative that outputs deterministic actions
    bounded by tanh to [-1, 1], then scaled to [0, 1].

    For exploration during training, we add noise externally.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super(TanhDeterministicActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Exploration noise std (decays during training)
        self.noise_std = 0.3

        # Build network
        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())
        layers.append(init_layer(nn.Linear(hidden_dim, action_dim), std=0.01))
        layers.append(nn.Tanh())  # Bound to [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass returning action in [0, 1].

        Tanh outputs [-1, 1], we scale to [0, 1].
        """
        tanh_out = self.network(state)  # [-1, 1]
        action = (tanh_out + 1.0) / 2.0  # [0, 1]
        return action

    def act(self, state, add_noise=True):
        """
        Get action with optional exploration noise.

        Returns:
            action: Action in [0, 1]
            log_prob: Dummy log_prob (not used in deterministic policy)
        """
        action = self.forward(state)

        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise
            action = torch.clamp(action, 0.0, 1.0)

        # Return dummy log_prob for compatibility
        log_prob = torch.zeros(action.shape[0], device=action.device)

        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """
        For deterministic policy, we use a pseudo-likelihood.

        Returns MSE-based pseudo log probability and zero entropy.
        """
        predicted_action = self.forward(state)

        # Pseudo log-prob based on squared error (for PPO compatibility)
        mse = ((predicted_action - action) ** 2).sum(dim=-1)
        pseudo_log_prob = -mse / (2 * self.noise_std ** 2)

        # No entropy for deterministic policy
        entropy = torch.zeros_like(pseudo_log_prob)

        return pseudo_log_prob, entropy

    def get_deterministic_action(self, state):
        """Get deterministic action for evaluation."""
        with torch.no_grad():
            return self.forward(state)

    def decay_noise(self, decay_rate=0.995, min_noise=0.05):
        """Decay exploration noise."""
        self.noise_std = max(self.noise_std * decay_rate, min_noise)


class GaussianActor(nn.Module):
    """
    Standard Gaussian Actor for MAPPO (most common in literature).

    Outputs mean and log_std for a Gaussian distribution over actions.
    Actions are sampled from N(mean, std) and then clipped to [0, 1].

    This is the standard approach used in most MAPPO papers.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super(GaussianActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared feature extractor
        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())

        self.feature_net = nn.Sequential(*layers)

        # Mean head
        self.mean_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)

        # Log std as learnable parameter (shared across all actions)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        """
        Forward pass returning mean and std.

        Returns:
            mean: Mean of Gaussian distribution
            std: Standard deviation (> 0)
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)

        # Apply sigmoid to bound mean to [0, 1]
        mean = torch.sigmoid(mean)

        # Ensure std is positive and not too small
        std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-6)

        return mean, std

    def act(self, state):
        """
        Sample action from Gaussian distribution.

        Returns:
            action: Sampled action in [0, 1]
            log_prob: Log probability of the action
        """
        mean, std = self.forward(state)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample action
        action = dist.sample()

        # Clip to [0, 1] for bounded action space
        action = torch.clamp(action, 0.0, 1.0)

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """
        Evaluate log probability and entropy for given state-action pairs.

        Args:
            state: Batch of states
            action: Batch of actions

        Returns:
            log_prob: Log probabilities
            entropy: Distribution entropy
        """
        mean, std = self.forward(state)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Compute log probability (before clipping)
        # Note: This is an approximation since we clip actions to [0,1]
        # but compute log_prob as if they're unbounded
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def get_deterministic_action(self, state):
        """Get deterministic action (mean) for evaluation."""
        with torch.no_grad():
            mean, _ = self.forward(state)
            return mean


class CentralizedCritic(nn.Module):
    """Centralized critic - shared across all agents, sees global state."""

    def __init__(self, global_state_dim, hidden_dim=64, num_layers=2):
        super(CentralizedCritic, self).__init__()

        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        layers.append(init_layer(nn.Linear(global_state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())
        layers.append(init_layer(nn.Linear(hidden_dim, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, global_state):
        return self.network(global_state)


# ============================================================
# 2. MAPPO CTDE CLASS (Beta Distribution)
# ============================================================

class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.actions = []
        self.states = []
        self.global_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def __len__(self):
        return len(self.states)


class MAPPO_CTDE:
    """
    Multi-Agent PPO with Centralized Training and Decentralized Execution.
    
    Uses Beta distribution policy for naturally bounded actions.

    - Each agent has its own decentralized Beta actor
    - All agents share a centralized critic (sees global state during training)
    - At execution time, each agent only uses its local actor with local observations
    """

    def __init__(self, num_agents, state_dim, global_state_dim, action_dim,
                 lr_actor, lr_critic,
                 actor_hidden_dim=64, critic_hidden_dim=64,
                 actor_num_layers=2, critic_num_layers=2,
                 policy_type='gaussian'):

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor_num_layers = actor_num_layers
        self.critic_num_layers = critic_num_layers
        self.policy_type = policy_type

        self.gamma = GAMMA_DISCOUNT
        self.gae_lambda = GAE_LAMBDA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS

        # Select actor class based on policy type
        if policy_type == 'beta':
            ActorClass = BetaActor
        elif policy_type == 'tanh':
            ActorClass = TanhDeterministicActor
        elif policy_type == 'gaussian':
            ActorClass = GaussianActor
        else:
            raise ValueError(f"Unknown policy type: {policy_type}. Choose from: 'beta', 'tanh', 'gaussian'")

        # Decentralized actors - one per agent
        self.actors = [
            ActorClass(state_dim, action_dim, actor_hidden_dim, actor_num_layers).to(device)
            for _ in range(num_agents)
        ]
        self.actors_old = [
            ActorClass(state_dim, action_dim, actor_hidden_dim, actor_num_layers).to(device)
            for _ in range(num_agents)
        ]
        for i in range(num_agents):
            self.actors_old[i].load_state_dict(self.actors[i].state_dict())

        # Centralized critic - shared by all agents
        self.critic = CentralizedCritic(global_state_dim, critic_hidden_dim, critic_num_layers).to(device)

        # Separate optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.MseLoss = nn.MSELoss()

    def select_action(self, agent_idx, state):
        """Select action for a specific agent using its old policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, logprob = self.actors_old[agent_idx].act(state_tensor)
        return action.cpu().numpy().flatten(), logprob

    def update(self, buffers):
        """Update all actors and the shared critic using collected experiences with GAE."""
        agent_data = []
        
        for agent_idx, (agent_id, buffer) in enumerate(sorted(buffers.items())):
            if len(buffer) == 0:
                continue

            # Convert buffer data to tensors
            old_states = torch.stack(buffer.states).detach().to(device)
            old_global_states = torch.stack(buffer.global_states).detach().to(device)
            old_actions = torch.stack(buffer.actions).detach().to(device)
            old_logprobs = torch.stack(buffer.logprobs).detach().to(device).squeeze()
            old_values = torch.stack(buffer.values).detach().to(device).squeeze()

            rewards = buffer.rewards
            is_terminals = buffer.is_terminals

            # Compute GAE advantages and returns
            advantages = []
            returns = []
            gae = 0

            # Process in reverse order
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

            agent_data.append({
                'agent_idx': agent_idx,
                'advantages': advantages,
                'returns': returns,
                'old_states': old_states,
                'old_global_states': old_global_states,
                'old_actions': old_actions,
                'old_logprobs': old_logprobs
            })

        if not agent_data:
            return

        # PPO update epochs
        for _ in range(self.K_epochs):
            for data in agent_data:
                agent_idx = data['agent_idx']
                advantages = data['advantages']
                returns = data['returns']
                old_states = data['old_states']
                old_global_states = data['old_global_states']
                old_actions = data['old_actions']
                old_logprobs = data['old_logprobs']

                # Evaluate actions with current policy
                logprobs, entropy = self.actors[agent_idx].evaluate(old_states, old_actions)
                logprobs = logprobs.squeeze()

                # Normalize advantages
                if len(advantages) > 1:
                    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    advantages_normalized = advantages

                # Compute policy ratio
                ratios = torch.exp(logprobs - old_logprobs)

                # Clipped surrogate objective
                surr1 = ratios * advantages_normalized
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_normalized

                # Actor loss (negative because we want to maximize)
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                # Update actor
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
                self.actor_optimizers[agent_idx].step()

                # Critic loss
                state_values = self.critic(old_global_states).squeeze()
                critic_loss = self.MseLoss(state_values, returns)

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        # Sync old policies
        for i in range(self.num_agents):
            self.actors_old[i].load_state_dict(self.actors[i].state_dict())

        # Decay noise for tanh policy
        if self.policy_type == 'tanh':
            for actor in self.actors:
                actor.decay_noise()
            for actor in self.actors_old:
                actor.noise_std = self.actors[0].noise_std

        # Clear buffers
        for buffer in buffers.values():
            buffer.clear()

    # ============================================================
    # MODEL SAVE / LOAD METHODS
    # ============================================================

    def save(self, filepath):
        """Save the model to a file."""
        save_dict = {
            'config': {
                'num_agents': self.num_agents,
                'state_dim': self.state_dim,
                'global_state_dim': self.global_state_dim,
                'action_dim': self.action_dim,
                'lr_actor': self.lr_actor,
                'lr_critic': self.lr_critic,
                'actor_hidden_dim': self.actor_hidden_dim,
                'critic_hidden_dim': self.critic_hidden_dim,
                'actor_num_layers': self.actor_num_layers,
                'critic_num_layers': self.critic_num_layers,
                'policy_type': self.policy_type,
            },
            'actors': [actor.state_dict() for actor in self.actors],
            'actors_old': [actor.state_dict() for actor in self.actors_old],
            'critic': self.critic.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

        if not filepath.endswith('.pt'):
            filepath = filepath + '.pt'

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device_to_use=None):
        """Load a model from a file."""
        if device_to_use is None:
            device_to_use = device

        if not filepath.endswith('.pt'):
            filepath = filepath + '.pt'

        checkpoint = torch.load(filepath, map_location=device_to_use)
        config = checkpoint['config']

        model = cls(
            num_agents=config['num_agents'],
            state_dim=config['state_dim'],
            global_state_dim=config['global_state_dim'],
            action_dim=config['action_dim'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            actor_hidden_dim=config['actor_hidden_dim'],
            critic_hidden_dim=config['critic_hidden_dim'],
            actor_num_layers=config['actor_num_layers'],
            critic_num_layers=config['critic_num_layers'],
            policy_type=config.get('policy_type', 'gaussian'),
        )

        for i, state_dict in enumerate(checkpoint['actors']):
            model.actors[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['actors_old']):
            model.actors_old[i].load_state_dict(state_dict)
        model.critic.load_state_dict(checkpoint['critic'])

        for i, state_dict in enumerate(checkpoint['actor_optimizers']):
            model.actor_optimizers[i].load_state_dict(state_dict)
        model.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        print(f"Model loaded from {filepath}")
        return model


# ============================================================
# 3. TRAINING LOGIC
# ============================================================

def run_marl_session(omega, seed, lr, episodes, num_classrooms=NUM_CLASSROOMS,
                     policy_type='gaussian', actor_hidden_dim=ACTOR_HIDDEN_DIM,
                     critic_hidden_dim=CRITIC_HIDDEN_DIM):
    """Runs a single MAPPO CTDE training session."""
    set_seed(seed)

    # TODO (Student): You will need to add a `shared_fraction` parameter to this function
    # (e.g., def run_mappo_training(..., shared_fraction=SHARED_FRACTION)).
    env = MultiClassroomEnv(
        num_classrooms=num_classrooms,
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        gamma=omega,
        continuous_action=True,
        cooperative_reward=COOPERATIVE_REWARD
    )

    num_agents = len(env.agents)
    state_dim = 2  # (normalized_infected, risk)
    global_state_dim = 2 * num_agents

    mappo = MAPPO_CTDE(
        num_agents=num_agents,
        state_dim=state_dim,
        global_state_dim=global_state_dim,
        action_dim=1,
        lr_actor=lr,
        lr_critic=lr,
        actor_hidden_dim=actor_hidden_dim,
        critic_hidden_dim=critic_hidden_dim,
        actor_num_layers=ACTOR_NUM_LAYERS,
        critic_num_layers=CRITIC_NUM_LAYERS,
        policy_type=policy_type
    )

    agent_ids = sorted(env.agents)
    buffers = {aid: RolloutBuffer() for aid in agent_ids}
    agent_idx_map = {aid: idx for idx, aid in enumerate(agent_ids)}

    reward_history = []
    time_step = 0

    for ep in range(episodes):
        obs = env.reset()
        n_obs = {aid: normalize_state(obs[aid], TOTAL_STUDENTS) for aid in env.agents}
        ep_reward = 0
        done = False

        while not done:
            time_step += 1
            
            # Construct global state
            g_state = np.concatenate([n_obs[aid] for aid in agent_ids])
            g_state_tensor = torch.tensor(g_state, dtype=torch.float32)
            
            actions_for_env = {}

            # Get value estimate for GAE
            with torch.no_grad():
                value = mappo.critic(g_state_tensor.unsqueeze(0).to(device))

            # Each agent selects action
            for aid in agent_ids:
                agent_idx = agent_idx_map[aid]
                s_t = torch.tensor(n_obs[aid], dtype=torch.float32, device=device)

                # Sample action from policy
                a_normalized, lp = mappo.select_action(agent_idx, n_obs[aid])
                a_normalized_tensor = torch.tensor(a_normalized, dtype=torch.float32, device=device)

                # Scale to environment action space
                a_env = a_normalized * TOTAL_STUDENTS
                actions_for_env[aid] = a_env

                # Store in buffer
                buffers[aid].states.append(s_t)
                buffers[aid].global_states.append(g_state_tensor)
                buffers[aid].actions.append(a_normalized_tensor)
                buffers[aid].logprobs.append(lp)
                buffers[aid].values.append(value.squeeze())

            # Environment step
            next_obs, rewards, dones, _ = env.step(actions_for_env)

            # Store rewards and terminals
            for aid in agent_ids:
                buffers[aid].rewards.append(rewards[aid])
                buffers[aid].is_terminals.append(dones[aid])
                n_obs[aid] = normalize_state(next_obs[aid], TOTAL_STUDENTS)

            ep_reward += sum(rewards.values()) / num_agents

            # Update policy periodically
            if time_step % UPDATE_TIMESTEP == 0:
                mappo.update(buffers)

            done = any(dones.values())

        reward_history.append(ep_reward)

    return mappo, reward_history


# ============================================================
# 4. POLICY EXTRACTION FUNCTIONS
# ============================================================

def extract_policy_grid(mappo_agent, agent_idx=0, total_students=TOTAL_STUDENTS,
                        grid_points=POLICY_GRID_POINTS):
    """Extracts a policy grid from a specific agent's actor."""
    infected_vals = np.linspace(0, total_students, grid_points)
    risk_vals = np.linspace(0, 1, grid_points)

    policy_grid = np.zeros((grid_points, grid_points), dtype=np.float32)

    for i, inf in enumerate(infected_vals):
        for j, risk in enumerate(risk_vals):
            s_norm = normalize_state([inf, risk], total_students)
            s_tensor = torch.FloatTensor(s_norm).to(device).unsqueeze(0)
            with torch.no_grad():
                action = mappo_agent.actors[agent_idx].get_deterministic_action(s_tensor).item()
            policy_grid[i, j] = action

    return policy_grid



# ============================================================
# 5. TUNING FUNCTIONS
# ============================================================

TUNE_EVAL_SEEDS = [101]


def evaluate_tuning_agents(omega_val, mappo_agent, eval_seeds):
    """Evaluates trained MAPPO_CTDE agent across multiple seeds."""
    total_rewards = []

    for seed in eval_seeds:
        set_seed(seed)
        eval_env = MultiClassroomEnv(
            num_classrooms=NUM_CLASSROOMS,
            total_students=TOTAL_STUDENTS,
            max_weeks=MAX_WEEKS,
            gamma=omega_val,
            continuous_action=True,
            cooperative_reward=COOPERATIVE_REWARD,
            eval_mode=False,
        )

        agent_ids = sorted(eval_env.agents)
        agent_idx_map = {aid: idx for idx, aid in enumerate(agent_ids)}

        obs = eval_env.reset()
        n_obs = {aid: normalize_state(obs[aid], TOTAL_STUDENTS) for aid in eval_env.agents}
        done = False
        total_reward = 0

        while not done:
            actions = {}
            for aid in agent_ids:
                agent_idx = agent_idx_map[aid]
                s_tensor = torch.FloatTensor(n_obs[aid]).to(device).unsqueeze(0)
                with torch.no_grad():
                    action_normalized = mappo_agent.actors[agent_idx].get_deterministic_action(s_tensor)
                    action_env = action_normalized.cpu().numpy().flatten() * TOTAL_STUDENTS
                actions[aid] = action_env

            next_obs, rewards, dones, _ = eval_env.step(actions)

            total_reward += sum(rewards.values()) / len(eval_env.agents)

            for aid in eval_env.agents:
                n_obs[aid] = normalize_state(next_obs[aid], TOTAL_STUDENTS)

            done = any(dones.values())

        total_rewards.append(total_reward)

    return np.mean(total_rewards)


def select_best_hyperparams(omega_results):
    """Selects the best hyperparameters based on reward."""
    if not omega_results:
        raise ValueError("No results to select from")

    sorted_results = sorted(omega_results, key=lambda r: -r['avg_eval_reward'])

    best_metrics = sorted_results[0]
    best_lr = best_metrics['lr']
    best_hidden_dim = best_metrics['hidden_dim']

    return (best_lr, best_hidden_dim), best_metrics, {}


def grid_search_tuning(policy_type='gaussian'):
    """Performs grid search tuning for each omega value."""
    optimized_hyperparams = {}

    print(f"\n--- Starting Grid Search Hyperparameter Tuning ({policy_type} policy) ---")
    print(f"Testing LRs: {LR_CANDIDATES}")
    print(f"Testing Hidden Dims: {HIDDEN_DIM_CANDIDATES}")
    start_time = time.time()

    for omega in OMEGA_VALUES:
        print(f"\n{'=' * 60}")
        print(f"*** Tuning for Omega = {omega} ***")
        print(f"{'=' * 60}")

        omega_results = []

        for lr in LR_CANDIDATES:
            for hidden_dim in HIDDEN_DIM_CANDIDATES:
                print(f"\n  Testing LR={lr:.6f}, Hidden Dim={hidden_dim}...")

                mappo, history = run_marl_session(omega, TUNE_SEED, lr, TUNE_EPISODES,
                                                  policy_type=policy_type,
                                                  actor_hidden_dim=hidden_dim,
                                                  critic_hidden_dim=hidden_dim)

                avg_reward = evaluate_tuning_agents(omega, mappo, TUNE_EVAL_SEEDS)

                result = {
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'avg_eval_reward': avg_reward,
                }
                omega_results.append(result)

                print(f"    Reward: {avg_reward:.2f}")

        best_hyperparams, best_metrics, _ = select_best_hyperparams(omega_results)
        best_lr, best_hidden_dim = best_hyperparams
        optimized_hyperparams[str(omega)] = {'lr': best_lr, 'hidden_dim': best_hidden_dim}

        print(f"\n  --> Best LR for Omega={omega}: {best_lr}, Best Hidden Dim: {best_hidden_dim}")
        print(f"      Reward: {best_metrics['avg_eval_reward']:.2f}")

    with open(HYPERPARAMS_FILE, 'w') as f:
        json.dump(optimized_hyperparams, f, indent=4)

    elapsed = time.time() - start_time
    print(f"\nTuning complete in {elapsed / 60:.1f} minutes")
    print(f"Optimized hyperparameters saved to {HYPERPARAMS_FILE}")

    return {float(k): v for k, v in optimized_hyperparams.items()}


def load_hyperparams():
    """Load optimized hyperparameters from file."""
    if not os.path.exists(HYPERPARAMS_FILE):
        print(f"WARNING: {HYPERPARAMS_FILE} not found. Using default LR=0.001, Hidden Dim=64")
        return {omega: {'lr': 0.001, 'hidden_dim': 64} for omega in OMEGA_VALUES}

    with open(HYPERPARAMS_FILE, 'r') as f:
        hyperparams = json.load(f)

    return {float(k): v for k, v in hyperparams.items()}


# ============================================================
# 6. FULL TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate_optimal(optimized_hyperparams, policy_type='gaussian'):
    """Runs full training with optimized hyperparameters and saves models."""
    print(f"\n--- Starting Full Training with Optimal Hyperparameters ({policy_type} MAPPO) ---")

    all_rewards_matrix = {}
    representative_agents = {}

    for omega in OMEGA_VALUES:
        hyperparams = optimized_hyperparams.get(omega, {'lr': 0.001, 'hidden_dim': 64})
        lr = hyperparams['lr']
        hidden_dim = hyperparams['hidden_dim']
        print(f"\n{'=' * 60}")
        print(f"*** Training Omega = {omega}, LR = {lr}, Hidden Dim = {hidden_dim} ***")
        print(f"{'=' * 60}")

        omega_rewards_runs = []

        for run in range(NUM_RUNS):
            seed = TUNE_SEED
            print(f"\n  Run {run + 1}/{NUM_RUNS} (seed={seed})")

            mappo, history = run_marl_session(omega, seed, lr, FULL_EPISODES,
                                              policy_type=policy_type,
                                              actor_hidden_dim=hidden_dim,
                                              critic_hidden_dim=hidden_dim)
            omega_rewards_runs.append(history)

            # Save model
            # TODO (Student): Update this path to include `num_classrooms` and `shared_fraction`.
            # Example: f"mappo_omega_{omega}_sf_{shared_fraction}_k_{num_classrooms}_hd_{hidden_dim}_run_{run}"
            # This prevents overwriting when you loop over different values!
            model_path = os.path.join(MODEL_DIR, f"mappo_omega_{omega}_hd_{hidden_dim}_run_{run}")
            mappo.save(model_path)

            if run == 0:
                representative_agents[omega] = mappo

        all_rewards_matrix[omega] = np.array(omega_rewards_runs)

    # Plot results
    plot_combined_rewards(all_rewards_matrix, NUM_RUNS)
    plot_policy_strips(representative_agents)

    print(f"\nTraining complete. Models saved to {MODEL_DIR}")

    return representative_agents


# ============================================================
# 7. PLOTTING FUNCTIONS
# ============================================================

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    HSV = [(x / n, 0.6, 0.95) for x in range(n)]
    RGB = [colorsys.hsv_to_rgb(*x) for x in HSV]
    return [
        '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b) in RGB
    ]


def plot_combined_rewards(all_rewards_matrix, num_runs):
    """Plots combined smoothed training rewards."""
    print("\n--- Generating Combined Rewards Plot ---")

    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    colors_lines = [cmap(i % 10) for i in range(len(OMEGA_VALUES))]

    def get_smoothed_data(data, window=100):
        kernel = np.ones(window) / window
        sliding_avg = np.convolve(data, kernel, mode='valid')
        growing_avg = np.cumsum(data[:window - 1]) / np.arange(1, window)
        return np.concatenate((growing_avg, sliding_avg))

    for idx, omega in enumerate(OMEGA_VALUES):
        data = all_rewards_matrix[omega]
        mean_rewards = np.mean(data, axis=0)
        std_rewards = np.std(data, axis=0)

        smoothed_mean = get_smoothed_data(mean_rewards, 100)
        smoothed_std = get_smoothed_data(std_rewards, 100)
        x_axis = np.arange(len(smoothed_mean))

        plt.plot(x_axis, smoothed_mean, label=f"$\\omega={omega}$", 
                 color=colors_lines[idx], linewidth=3)
        plt.fill_between(x_axis, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                         color=colors_lines[idx], alpha=0.2)

    plt.xlim(left=0)
    plt.margins(x=0)
    plt.xlabel("Episode", fontsize=14, fontweight='bold')
    plt.ylabel("Reward", fontsize=14, fontweight='bold')
    plt.title("MAPPO (CTDE Beta) Training Reward", fontsize=14, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(OMEGA_VALUES), borderaxespad=0., fontsize=12, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_mappo_rewards_ci.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_policy_strips(representative_agents):
    """Plots the optimal policy as a heatmap strip for each omega and each agent."""
    print("\n--- Generating Optimal Policy Strips Plot ---")

    first_agent = list(representative_agents.values())[0]
    num_agents = first_agent.num_agents

    fig, axes = plt.subplots(num_agents, len(OMEGA_VALUES),
                             figsize=(20, 3.5 * num_agents), sharey=True)

    if num_agents == 1:
        axes = axes.reshape(1, -1)

    base_colors = generate_distinct_colors(12)
    continuous_cmap = LinearSegmentedColormap.from_list("custom_hsv_continuous", base_colors, N=256)

    for idx, omega in enumerate(OMEGA_VALUES):
        mappo_agent = representative_agents[omega]

        for agent_idx in range(num_agents):
            ax = axes[agent_idx, idx]
            policy_grid = extract_policy_grid(mappo_agent, agent_idx)

            im = ax.imshow(policy_grid, extent=[0, 1, 0, TOTAL_STUDENTS],
                           origin="lower", aspect="auto", cmap=continuous_cmap,
                           vmin=0.0, vmax=1.0)

            ax.set_title(f"$\\omega={omega}$ | Agent {agent_idx}")

            if agent_idx == num_agents - 1:
                ax.set_xlabel("Risk")
            if idx == 0:
                ax.set_ylabel("Infected Count")
            else:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Action (Capacity Fraction)', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_mappo_optimal_policies.png"),
                dpi=300, bbox_inches='tight')
    plt.close()



# ============================================================
# 8. MAIN CONTROL FUNCTION
# ============================================================

def main(mode='train', policy_type='gaussian'):
    """
    Main function to control execution flow.

    Modes:
    - 'tune': Runs Grid Search to find the optimal hyperparameters (LR and Hidden Dim) for each omega and saves them.
    - 'train': Loads optimal hyperparameters and runs full training, evaluation, and plotting.
    - 'tune_and_train': Runs tuning, then immediately runs training.

    Policy types:
    - 'gaussian': Standard Gaussian policy (most common MAPPO, recommended)
    - 'beta': Beta distribution (naturally bounded to [0,1])
    - 'tanh': Tanh deterministic with exploration noise
    """
    print(f"Policy: {policy_type} MAPPO")

    optimized_hyperparams = {}

    if mode == 'tune' or mode == 'tune_and_train':
        optimized_hyperparams = grid_search_tuning(policy_type=policy_type)
    else:
        optimized_hyperparams = load_hyperparams()

    if mode == 'train' or mode == 'tune_and_train':
        train_and_evaluate_optimal(optimized_hyperparams, policy_type=policy_type)

    print(f"\nAll processing complete. Results in {OUTPUT_DIR}")


if __name__ == '__main__':
    # TODO (Student): Import `argparse` at the top of the file.
    # Set up argparse here to accept:
    # --num_classrooms (int, default=NUM_CLASSROOMS)
    # --shared_fraction (float, default=SHARED_FRACTION)
    # Then pass those parsed arguments into main()!
    main(mode='tune_and_train', policy_type='beta')

    # To use other policies:
    # main(mode='tune_and_train', policy_type='beta')
    # main(mode='tune_and_train', policy_type='tanh')