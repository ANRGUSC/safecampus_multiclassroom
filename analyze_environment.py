"""
Diagnostic Evaluation for Multi-Classroom Epidemic Control

The methods do NOT solve the same problem, so this is deliberately framed as a
DIAGNOSTIC ("what is each method doing, and why") rather than a leaderboard.
Methods are grouped into matched information regimes and compared WITHIN a regime;
cross-regime differences are reported as named, caveated decompositions.

Information regimes
-------------------
  Full-info  (acts on the full joint state):
      - DP             : perfect-foresight optimum (discrete-action ceiling)
      - Joint-Myopic   : full-state 1-step greedy
      - Centralized PPO: learned joint controller
  Local-info (acts on local observation only):
      - Dec-Myopic     : per-room 1-step greedy on its OWN state (neighbors unseen)
      - CTDE MAPPO     : learned decentralized actors
  Floor:
      - Random

Named decompositions (each isolates one factor)
  price of decentralization = Centralized - CTDE        (cost of local-only obs)
  optimality / foresight gap = DP - Centralized          (within full-info: fair)
  value of lookahead         = Centralized - Joint-Myopic

Scenario families (both reported, paired across methods via shared seeds)
  synthetic : K held-out seeds -> sampled sine risk + random init (in-distribution)
  real      : CSV weekly risk path fixed, random init varied over the K seeds

All metrics are reward-derived. CIs / significance use bootstrap (no scipy needed).
The environment is the SAME for every method (TOTAL_STUDENTS, dynamics, reward).

Author: SafeCampus Project
"""

# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import os
import json
import time
import pandas as pd
import argparse

# pyrefly: ignore [missing-import]
from environment.multiclassroom import MultiClassroomEnv

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared environment config (must match the trainers)
TOTAL_STUDENTS = 50
NUM_CLASSROOMS = 2
MAX_WEEKS = 15
COOPERATIVE_REWARD = True
COMMUNITY_RISK_FILE = "weekly_risk_sample_b.csv"

# Discrete action grid (DP + both myopic baselines, for fair comparison)
N_ACTION_BINS = 7

# Paired evaluation: K scenarios per family, shared across all methods.
# DP is run on a smaller subset (its cache is per (omega, scenario) -> expensive).
K_SCENARIOS = 50
DP_SCENARIOS = 10
EVAL_SEED_BASE = 9000  # held-out block, disjoint from training seeds (123, 101, 42)

# Default Sweep parameters
OMEGA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Default values
SHARED_FRACTION = 0.3
NUM_CLASSROOMS = 2

# Shared configs (MUST MATCH TRAINING)
TOTAL_STUDENTS = 50

# Bootstrap
N_BOOT = 2000

# Method registry: name -> (label, league, color)
LEAGUE = {
    'dp':            ('DP (foresight ceiling)', 'full_info', 'gold'),
    'joint_myopic':  ('Joint-Myopic',           'full_info', 'forestgreen'),
    'centralized':   ('Centralized PPO',         'full_info', 'steelblue'),
    'dec_myopic':    ('Dec-Myopic',              'local_info', 'darkorange'),
    'ctde':          ('CTDE MAPPO',              'local_info', 'coral'),
    'random':        ('Random',                  'floor', 'gray'),
}
FULL_INFO = ['dp', 'joint_myopic', 'centralized']
LOCAL_INFO = ['dec_myopic', 'ctde']
ALL_METHODS = ['dp', 'joint_myopic', 'centralized', 'dec_myopic', 'ctde', 'random']

plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})


# ============================================================
# SCENARIOS
# ============================================================

def load_real_risk(path=COMMUNITY_RISK_FILE):
    """Load the real weekly community-risk series (truncated/padded to MAX_WEEKS)."""
    if not os.path.exists(path):
        print(f"  WARNING: real risk file {path} not found; real family will be skipped.")
        return None
    df = pd.read_csv(path)
    risk_col = next((c for c in df.columns if 'risk' in c.lower()), df.columns[-1])
    risk = [float(x) for x in df[risk_col].tolist()]
    if len(risk) >= MAX_WEEKS:
        return risk[:MAX_WEEKS]
    return risk + [risk[-1]] * (MAX_WEEKS - len(risk))  # pad with last value


def build_scenarios(k=K_SCENARIOS):
    """Build the shared, paired scenario sets for both families."""
    seeds = [EVAL_SEED_BASE + i for i in range(k)]
    real_risk = load_real_risk()
    families = {'synthetic': {'seeds': seeds, 'risk': None}}
    if real_risk is not None:
        families['real'] = {'seeds': seeds, 'risk': real_risk}
    return families


def make_env(omega):
    return MultiClassroomEnv(
        num_classrooms=NUM_CLASSROOMS,
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        gamma=omega,
        continuous_action=True,
        cooperative_reward=COOPERATIVE_REWARD,
        eval_mode=False,
        # (Otherwise python will crash because `shared_fraction` lowercase doesn't exist inside this function)
        shared_fraction=SHARED_FRACTION
    )


def current_risk(env):
    w = min(env.current_week, len(env.shared_community_risk) - 1)
    return env.shared_community_risk[w]


# ============================================================
# POLICIES  (interface: act(env, obs, agent_ids, omega) -> {aid: np.array([value])})
# ============================================================

class RandomPolicy:
    name = 'random'

    def available(self):
        return True

    def act(self, env, obs, agent_ids, omega):
        return {aid: np.array([np.random.rand() * env.total_students]) for aid in agent_ids}


class JointMyopicPolicy:
    """Full-state 1-step greedy over a discrete joint action grid (full-info regime)."""
    name = 'joint_myopic'

    def __init__(self, n_action_bins=N_ACTION_BINS):
        from itertools import product
        self.grid = np.linspace(0, TOTAL_STUDENTS, n_action_bins)
        self.combos = list(product(self.grid, repeat=NUM_CLASSROOMS))
        self._scratch = make_env(0.5)  # reused; gamma set per-call below

    def available(self):
        return True

    def _scratch_at(self, env, omega):
        s = self._scratch
        s.gamma = omega
        s.student_status = list(env.student_status)
        s.current_week = env.current_week
        s.shared_community_risk = list(env.shared_community_risk)
        return s

    def act(self, env, obs, agent_ids, omega):
        best_r, best_a = -np.inf, None
        for combo in self.combos:
            s = self._scratch_at(env, omega)
            acts = {aid: np.array([combo[i]]) for i, aid in enumerate(agent_ids)}
            _, rewards, _, _ = s.step(acts)
            r = rewards[agent_ids[0]]  # cooperative -> shared
            if r > best_r:
                best_r, best_a = r, combo
        return {aid: np.array([best_a[i]]) for i, aid in enumerate(agent_ids)}


class DecentralizedMyopicPolicy:
    """
    Per-room 1-step greedy on the room's OWN state only (local-info regime).

    Each room cannot see its neighbors, so it optimizes against a single-classroom
    model of itself (no coupling). The independently chosen actions are then applied
    to the true coupled environment. This is the matched in-regime baseline for CTDE.
    """
    name = 'dec_myopic'

    def __init__(self, n_action_bins=N_ACTION_BINS):
        self.grid = np.linspace(0, TOTAL_STUDENTS, n_action_bins)
        self._sim = MultiClassroomEnv(
            num_classrooms=1, total_students=TOTAL_STUDENTS, max_weeks=MAX_WEEKS,
            gamma=0.5, continuous_action=True, cooperative_reward=True, eval_mode=False, shared_fraction=SHARED_FRACTION)

    def available(self):
        return True

    def act(self, env, obs, agent_ids, omega):
        risk = current_risk(env)
        self._sim.gamma = omega
        self._sim.shared_community_risk = [risk] * MAX_WEEKS
        actions = {}
        for i, aid in enumerate(agent_ids):
            infected_i = env.student_status[i]
            best_r, best_a = -np.inf, 0.0
            for a in self.grid:
                self._sim.student_status = [infected_i]
                self._sim.current_week = 0
                _, rewards, _, _ = self._sim.step({'classroom_0': np.array([a])})
                r = rewards['classroom_0']
                if r > best_r:
                    best_r, best_a = r, a
            actions[aid] = np.array([best_a])
        return actions


class CentralizedPolicy:
    """Trained centralized controller (full-info regime)."""
    name = 'centralized'

    def __init__(self, omega):
        self.model = _load_centralized(omega)

    def available(self):
        return self.model is not None

    def act(self, env, obs, agent_ids, omega):
        # pyrefly: ignore [missing-import]
        import torch
        gstate = []
        for aid in agent_ids:
            gstate.append(obs[aid][0] / env.total_students)
            gstate.append(obs[aid][1])
        device = next(self.model.actor.parameters()).device
        st = torch.tensor(np.array(gstate, dtype=np.float32), device=device).unsqueeze(0)
        with torch.no_grad():
            ja = self.model.actor.get_deterministic_action(st).cpu().numpy().flatten()
        return {aid: np.array([ja[i] * env.total_students]) for i, aid in enumerate(agent_ids)}


class CTDEPolicy:
    """Trained decentralized actors (local-info regime)."""
    name = 'ctde'

    def __init__(self, omega):
        self.model, self.normalize = _load_ctde(omega)

    def available(self):
        return self.model is not None

    def act(self, env, obs, agent_ids, omega):
        # pyrefly: ignore [missing-import]
        import torch
        actions = {}
        for i, aid in enumerate(agent_ids):
            s = self.normalize(obs[aid], env.total_students)
            device = next(self.model.actors[i].parameters()).device
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = self.model.actors[i].get_deterministic_action(st)
            actions[aid] = a.cpu().numpy().flatten() * env.total_students
        return actions


# ---- trained-model loaders (graceful: return None if not found) ----

# Awesome, you perfectly updated the path resolution!
def _resolve_model_path(results_dir, prefix, omega, shared_fraction=SHARED_FRACTION, num_classrooms=NUM_CLASSROOMS):
    model_dir = os.path.join(results_dir, "models")
    if not os.path.exists(model_dir):
        return None
        
    # Search for the newly named models: {prefix}_omega_{omega}_sf_{shared_fraction}_k_{num_classrooms}_hd_{anything}_run_0.pt
    search_prefix = f"{prefix}_omega_{omega}_sf_{shared_fraction}_k_{num_classrooms}_hd_"
    for filename in os.listdir(model_dir):
        if filename.startswith(search_prefix) and filename.endswith("_run_0.pt"):
            return os.path.join(model_dir, filename[:-3])  # Strip the .pt extension for the loader

    # Fallback to the old Phase 1 naming scheme if the new one isn't found
    p_old = os.path.join(model_dir, f"{prefix}_omega_{omega}_run_0")
    return p_old if os.path.exists(p_old + '.pt') else None


def _load_centralized(omega):
    try:
        # pyrefly: ignore [missing-import]
        from ppo_centralized import CentralizedPPO
        path = _resolve_model_path("centralized_ppo_results", "centralized", omega, shared_fraction=SHARED_FRACTION, num_classrooms=NUM_CLASSROOMS)
        if path is None:
            return None
        return CentralizedPPO.load(path)
    except Exception as e:
        print(f"    Centralized load failed (omega={omega}): {e}")
        return None


def _load_ctde(omega):
    try:
        import importlib
        mod = None
        for name in ("ppo_ctde", "mappo_ctde_beta"):
            try:
                mod = importlib.import_module(name)
                break
            except ImportError:
                continue
        if mod is None:
            return None, None
        path = _resolve_model_path("mappo_results", "mappo", omega, shared_fraction=SHARED_FRACTION, num_classrooms=NUM_CLASSROOMS)
        if path is None:
            return None, None
        return mod.MAPPO_CTDE.load(path), mod.normalize_state
    except Exception as e:
        print(f"    CTDE load failed (omega={omega}): {e}")
        return None, None


# ============================================================
# DP UPPER BOUND (per-scenario, perfect foresight of the risk path)
# ============================================================

class DPUpperBound:
    """On-demand discrete-action DP with memoization, for a FIXED risk trajectory."""
    name = 'dp'

    def __init__(self, omega, risk_vector, num_classrooms=NUM_CLASSROOMS,
                 total_students=TOTAL_STUDENTS, max_weeks=MAX_WEEKS, n_action_bins=N_ACTION_BINS):
        from itertools import product
        self.omega = omega
        self.risk_vector = list(risk_vector)
        self.num_classrooms = num_classrooms
        self.max_weeks = max_weeks
        self.action_vals = np.linspace(0, total_students, n_action_bins)
        self.combos = list(product(self.action_vals, repeat=self.num_classrooms))
        self.value_cache = {}
        self.policy_cache = {}
        self._sim = MultiClassroomEnv(
            num_classrooms=num_classrooms, total_students=total_students, max_weeks=max_weeks,
            gamma=omega, continuous_action=True, cooperative_reward=True, eval_mode=False)
        self._aids = sorted(self._sim.agents)

    def available(self):
        return True

    def _simulate_step(self, infected, actions, week):
        self._sim.student_status = list(infected)
        self._sim.current_week = week
        self._sim.shared_community_risk = self.risk_vector
        acts = {self._aids[i]: np.array([actions[i]]) for i in range(self.num_classrooms)}
        _, rewards, _, _ = self._sim.step(acts)
        return list(self._sim.student_status), rewards[self._aids[0]]

    def get_value(self, state, t):
        if t >= self.max_weeks:
            return 0.0
        key = (*state, t)
        if key in self.value_cache:
            return self.value_cache[key]
        best_v, best_a = -np.inf, None
        for combo in self.combos:
            nxt, r = self._simulate_step(list(state), combo, t)
            q = r + self.get_value(tuple(nxt), t + 1)
            if q > best_v:
                best_v, best_a = q, combo
        self.value_cache[key] = best_v
        self.policy_cache[key] = best_a
        return best_v

    def act(self, env, obs, agent_ids, omega):
        state = tuple(int(round(x)) for x in env.student_status)
        t = env.current_week
        if (*state, t) not in self.value_cache:
            self.get_value(state, t)
        a = self.policy_cache[(*state, t)]
        return {aid: np.array([a[i]]) for i, aid in enumerate(agent_ids)}


# ============================================================
# ROLLOUT
# ============================================================

def run_episode(omega, scenario_seed, risk_override, policy):
    """Run one deterministic-policy episode; returns reward + per-week diagnostics."""
    env = make_env(omega)
    obs = env.reset(seed=scenario_seed, risk_override=risk_override)
    agent_ids = sorted(env.agents)

    total_reward = 0.0
    admitted, infected = [], []
    done = False
    while not done:
        actions = policy.act(env, obs, agent_ids, omega)
        obs, rewards, dones, _ = env.step(actions)
        total_reward += rewards[agent_ids[0]]  # cooperative -> shared
        admitted.append(float(sum(env.allowed_students)))
        infected.append(float(sum(env.student_status)))
        done = any(dones.values())

    return {'reward': total_reward, 'admitted': admitted, 'infected': infected}


# ============================================================
# STATISTICS (bootstrap, no scipy)
# ============================================================

def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        m = float(values.mean()) if len(values) else 0.0
        return m, m, m
    idx = np.random.randint(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap(a_by_seed, b_by_seed, n_boot=N_BOOT):
    """Paired diff (a - b) over shared seeds: mean, 95% CI, two-sided bootstrap p."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 2:
        return None
    diffs = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds], dtype=float)
    idx = np.random.randint(0, len(diffs), size=(n_boot, len(diffs)))
    bmeans = diffs[idx].mean(axis=1)
    p = 2.0 * min((bmeans <= 0).mean(), (bmeans >= 0).mean())
    return {
        'mean': float(diffs.mean()),
        'ci_lo': float(np.percentile(bmeans, 2.5)),
        'ci_hi': float(np.percentile(bmeans, 97.5)),
        'p_value': float(min(1.0, p)),
        'n': len(seeds),
    }


# ============================================================
# EVALUATION DRIVER
# ============================================================

def evaluate(families, omegas=OMEGA_VALUES, k=K_SCENARIOS, dp_k=DP_SCENARIOS):
    """
    Returns results[family][omega][method] = {
        'by_seed': {seed: reward}, 'rewards': [...], 'mean','ci_lo','ci_hi',
        'admitted': mean weekly vector, 'infected': mean weekly vector,
        'cum_admitted': float, 'cum_infected': float,
    }
    """
    results = {}
    for fam, spec in families.items():
        seeds = spec['seeds'][:k]
        risk = spec['risk']
        results[fam] = {}
        print(f"\n{'#' * 70}\nFAMILY: {fam}  (K={len(seeds)} scenarios)\n{'#' * 70}")

        for omega in omegas:
            print(f"\n=== omega = {omega} ===")
            results[fam][omega] = {}

            # Build policies once per omega (models loaded once)
            policies = {
                'random': RandomPolicy(),
                'joint_myopic': JointMyopicPolicy(),
                'dec_myopic': DecentralizedMyopicPolicy(),
                'centralized': CentralizedPolicy(omega),
                'ctde': CTDEPolicy(omega),
            }

            for mname, policy in policies.items():
                if not policy.available():
                    print(f"  {LEAGUE[mname][0]:<24}: model not found, skipped")
                    results[fam][omega][mname] = None
                    continue
                results[fam][omega][mname] = _eval_method(omega, mname, policy, seeds, risk)

            # DP: per-scenario solve on a subset (expensive)
            dp_seeds = seeds[:dp_k]
            by_seed, traj_adm, traj_inf = {}, [], []
            t0 = time.time()
            for s in dp_seeds:
                # DP needs the exact risk path of this scenario
                scen_risk = risk if risk is not None else _scenario_risk(s)
                dp = DPUpperBound(omega, scen_risk, num_classrooms=NUM_CLASSROOMS)
                out = run_episode(omega, s, risk, dp)
                by_seed[s] = out['reward']
                traj_adm.append(out['admitted'])
                traj_inf.append(out['infected'])
            results[fam][omega]['dp'] = _summarize(by_seed, traj_adm, traj_inf)
            print(f"  {LEAGUE['dp'][0]:<24}: {results[fam][omega]['dp']['mean']:.1f} "
                  f"(subset n={len(dp_seeds)}, {time.time() - t0:.0f}s)")

    return results


def _scenario_risk(seed):
    """Recover the synthetic risk path a given seed produces (for DP foresight)."""
    env = make_env(0.5)
    env.reset(seed=seed)
    return list(env.shared_community_risk)


def _eval_method(omega, mname, policy, seeds, risk):
    by_seed, traj_adm, traj_inf = {}, [], []
    for s in seeds:
        out = run_episode(omega, s, risk, policy)
        by_seed[s] = out['reward']
        traj_adm.append(out['admitted'])
        traj_inf.append(out['infected'])
    
    summary = _summarize(by_seed, traj_adm, traj_inf)
    
    print(f"  {LEAGUE[mname][0]:<24}: {summary['mean']:.1f} "
          f"[{summary['ci_lo']:.1f}, {summary['ci_hi']:.1f}]")
          
    return summary


def _summarize(by_seed, traj_adm, traj_inf):
    rewards = list(by_seed.values())
    mean, lo, hi = bootstrap_ci(rewards)
    adm = np.array(traj_adm, dtype=float)
    inf = np.array(traj_inf, dtype=float)
    return {
        'by_seed': by_seed,
        'rewards': rewards,
        'mean': mean, 'ci_lo': lo, 'ci_hi': hi,
        'admitted': adm.mean(axis=0).tolist(),
        'infected': inf.mean(axis=0).tolist(),
        'cum_admitted': float(adm.sum(axis=1).mean()),
        'cum_infected': float(inf.sum(axis=1).mean()),
    }


# ============================================================
# DECOMPOSITIONS
# ============================================================

def compute_decompositions(results):
    """Named, paired cross/within-regime gaps per family/omega."""
    pairs = {
        'price_of_decentralization': ('centralized', 'ctde'),   # cross-regime
        'optimality_gap':            ('dp', 'centralized'),     # within full-info
        'value_of_lookahead':        ('centralized', 'joint_myopic'),
        'ctde_vs_dec_myopic':        ('ctde', 'dec_myopic'),    # within local-info
    }
    decomp = {}
    for fam in results:
        decomp[fam] = {}
        for omega in results[fam]:
            decomp[fam][omega] = {}
            for label, (a, b) in pairs.items():
                ra = results[fam][omega].get(a)
                rb = results[fam][omega].get(b)
                if ra is None or rb is None:
                    decomp[fam][omega][label] = None
                    continue
                decomp[fam][omega][label] = paired_bootstrap(ra['by_seed'], rb['by_seed'])
    return decomp


# ============================================================
# PLOTS
# ============================================================

def _omega_x(omegas):
    return np.arange(len(omegas)), [f'{o}' for o in omegas]


def plot_rewards_by_league(results, fam, omegas=OMEGA_VALUES):
    x, xlabels = _omega_x(omegas)
    fig, ax = plt.subplots(figsize=(10, 6))
    for mname in ALL_METHODS:
        label, league, color = LEAGUE[mname]
        means, los, his, xs = [], [], [], []
        for i, o in enumerate(omegas):
            r = results[fam][o].get(mname)
            if r is None:
                continue
            xs.append(i); means.append(r['mean']); los.append(r['ci_lo']); his.append(r['ci_hi'])
        if not xs:
            continue
        ls = '-' if league == 'full_info' else ('--' if league == 'local_info' else ':')
        ax.plot(xs, means, ls, marker='o', color=color, label=label, linewidth=2.2)
        ax.fill_between(xs, los, his, color=color, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_xlabel('Omega (ω)'); ax.set_ylabel('Episode Reward')
    ax.set_title(f'Reward by Method — {fam} scenarios\n'
                 'solid = full-info, dashed = local-info, dotted = floor', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"rewards_by_league_{fam}_k_{NUM_CLASSROOMS}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalized_score(results, fam, omegas=OMEGA_VALUES):
    """Fraction of full-info optimal recovered: (return - Random)/(DP - Random)."""
    x, xlabels = _omega_x(omegas)
    fig, ax = plt.subplots(figsize=(10, 6))
    for mname in ['centralized', 'joint_myopic', 'ctde', 'dec_myopic']:
        label, league, color = LEAGUE[mname]
        xs, ys = [], []
        for i, o in enumerate(omegas):
            r = results[fam][o].get(mname)
            dp = results[fam][o].get('dp')
            rnd = results[fam][o].get('random')
            if r is None or dp is None or rnd is None:
                continue
            denom = dp['mean'] - rnd['mean']
            if abs(denom) < 1e-9:
                continue
            xs.append(i); ys.append((r['mean'] - rnd['mean']) / denom)
        if not xs:
            continue
        ls = '-' if league == 'full_info' else '--'
        ax.plot(xs, ys, ls, marker='o', color=color, label=label, linewidth=2.2)
    ax.axhline(1.0, color='gold', linestyle=':', alpha=0.7, label='DP optimal')
    ax.axhline(0.0, color='gray', linestyle=':', alpha=0.7, label='Random floor')
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_xlabel('Omega (ω)'); ax.set_ylabel('Fraction of full-info optimal')
    ax.set_title(f'Fraction of full-info optimal recovered — {fam}\n'
                 '(local-info shortfall includes the price of decentralization)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"normalized_score_{fam}_k_{NUM_CLASSROOMS}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_decomposition(decomp, fam, omegas=OMEGA_VALUES):
    x, xlabels = _omega_x(omegas)
    labels = {
        'price_of_decentralization': ('Price of decentralization (Cent - CTDE)', 'crimson'),
        'optimality_gap':            ('Optimality gap (DP - Cent)', 'goldenrod'),
        'value_of_lookahead':        ('Value of lookahead (Cent - Myopic)', 'teal'),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (label, color) in labels.items():
        xs, means, errlo, errhi = [], [], [], []
        for i, o in enumerate(omegas):
            d = decomp[fam][o].get(key)
            if d is None:
                continue
            xs.append(i); means.append(d['mean'])
            errlo.append(d['mean'] - d['ci_lo']); errhi.append(d['ci_hi'] - d['mean'])
        if not xs:
            continue
        ax.errorbar(xs, means, yerr=[errlo, errhi], marker='o', color=color,
                    label=label, linewidth=2.2, capsize=3)
    ax.axhline(0.0, color='black', linewidth=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_xlabel('Omega (ω)'); ax.set_ylabel('Reward difference')
    ax.set_title(f'Reward decompositions (paired, 95% CI) — {fam}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"decomposition_{fam}_k_{NUM_CLASSROOMS}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_reward_terms(results, fam, omegas=OMEGA_VALUES):
    """The two reward terms vs omega: cumulative admitted (utility) and infected (cost)."""
    x, xlabels = _omega_x(omegas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for term, ax, ylab in [('cum_admitted', axes[0], 'Cumulative admitted (utility)'),
                           ('cum_infected', axes[1], 'Cumulative infected (cost)')]:
        for mname in ALL_METHODS:
            label, league, color = LEAGUE[mname]
            xs, ys = [], []
            for i, o in enumerate(omegas):
                r = results[fam][o].get(mname)
                if r is None:
                    continue
                xs.append(i); ys.append(r[term])
            if not xs:
                continue
            ls = '-' if league == 'full_info' else ('--' if league == 'local_info' else ':')
            ax.plot(xs, ys, ls, marker='o', color=color, label=label, linewidth=2)
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
        ax.set_xlabel('Omega (ω)'); ax.set_ylabel(ylab)
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best', fontsize=9)
    fig.suptitle(f'Reward terms vs ω — {fam}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"reward_terms_{fam}_k_{NUM_CLASSROOMS}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_trajectories(results, fam, omegas=OMEGA_VALUES):
    """Weekly admitted / infected curves (averaged over scenarios), per method, per ω."""
    weeks = np.arange(MAX_WEEKS)
    fig, axes = plt.subplots(2, len(omegas), figsize=(3.3 * len(omegas), 6.5), sharex=True)
    if len(omegas) == 1:
        axes = axes.reshape(2, 1)
    for col, o in enumerate(omegas):
        for mname in ALL_METHODS:
            label, league, color = LEAGUE[mname]
            r = results[fam][o].get(mname)
            if r is None:
                continue
            ls = '-' if league == 'full_info' else ('--' if league == 'local_info' else ':')
            axes[0, col].plot(weeks[:len(r['admitted'])], r['admitted'], ls, color=color, linewidth=1.8)
            axes[1, col].plot(weeks[:len(r['infected'])], r['infected'], ls, color=color,
                              linewidth=1.8, label=label)
        axes[0, col].set_title(f'ω={o}')
        axes[1, col].set_xlabel('Week')
        if col == 0:
            axes[0, col].set_ylabel('Admitted (sum)')
            axes[1, col].set_ylabel('Infected (sum)')
        for r_ in (0, 1):
            axes[r_, col].grid(True, linestyle='--', alpha=0.4)
    handles, labels_ = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=len(ALL_METHODS), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'Behavior over time — {fam}  (admitted = utility, infected = cost)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, f"trajectories_{fam}_k_{NUM_CLASSROOMS}.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# REPORTING
# ============================================================

def save_outputs(results, decomp, omegas=OMEGA_VALUES):
    # CSV table
    rows = []
    for fam in results:
        for o in omegas:
            row = {'family': fam, 'omega': o}
            for m in ALL_METHODS:
                r = results[fam][o].get(m)
                row[f'{m}_mean'] = r['mean'] if r else np.nan
                row[f'{m}_ci_lo'] = r['ci_lo'] if r else np.nan
                row[f'{m}_ci_hi'] = r['ci_hi'] if r else np.nan
            for key in ('price_of_decentralization', 'optimality_gap', 'value_of_lookahead'):
                d = decomp[fam][o].get(key)
                row[f'{key}_mean'] = d['mean'] if d else np.nan
                row[f'{key}_p'] = d['p_value'] if d else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f"diagnostic_results_k_{NUM_CLASSROOMS}.csv"), index=False)

    # JSON (drop per-seed/per-week bulk; keep summaries + decompositions)
    out = {}
    for fam in results:
        out[fam] = {}
        for o in omegas:
            out[fam][str(o)] = {'methods': {}, 'decomp': {}}
            for m in ALL_METHODS:
                r = results[fam][o].get(m)
                out[fam][str(o)]['methods'][m] = None if r is None else {
                    'mean': r['mean'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
                    'cum_admitted': r['cum_admitted'], 'cum_infected': r['cum_infected'],
                }
            for key, d in decomp[fam][o].items():
                out[fam][str(o)]['decomp'][key] = d
    with open(os.path.join(OUTPUT_DIR, f"diagnostic_results_k_{NUM_CLASSROOMS}.json"), 'w') as f:
        json.dump(out, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE — results in {OUTPUT_DIR}/")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diagnostic evaluation for multi-classroom environment.")
    parser.add_argument("--num_classrooms", type=int, default=NUM_CLASSROOMS, help="Number of classrooms (K)")
    args = parser.parse_args()

    # Update global variable so it's used dynamically everywhere
    NUM_CLASSROOMS = args.num_classrooms

    print(f"Starting diagnostic evaluation for K={NUM_CLASSROOMS} classrooms...")
    families = build_scenarios()
    
    # Run evaluation
    results = evaluate(families)
    
    # Compute decompositions
    decomp = compute_decompositions(results)
    
    # Generate all plots
    for fam in families:
        plot_rewards_by_league(results, fam)
        plot_normalized_score(results, fam)
        plot_decomposition(decomp, fam)
        plot_reward_terms(results, fam)
        plot_trajectories(results, fam)
        
    # Save CSV and JSON
    save_outputs(results, decomp)