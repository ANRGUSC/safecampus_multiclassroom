# SafeCampus MultiClassroom

Reinforcement learning for **epidemic-aware classroom capacity control**. Each week a
controller decides how many students each classroom may admit, trading attendance (utility)
against infections (cost) under a time-varying community risk. We study two RL controllers
(centralized and CTDE) against optimal and myopic references.

> The project is framed as an **RL/MARL methods study** that decomposes the difficulty of the
> control problem into **foresight**, **coordination/observability**, and **scale**, each
> measured against a perfect-foresight DP optimum. See **`EXPERIMENT_PLAN.docx`** for the full
> research questions, hypotheses, and experiment matrix.

## Problem

For `K` classrooms over `MAX_WEEKS` weeks, the per-room observation is `(infected, community
risk)` where **risk is a shared exogenous signal**. Classrooms couple **only through shared
students** (`shared_fraction`). The cooperative reward trades utility vs cost via preference ω:

```
r_i = ω · allowed_i − (1 − ω) · infected_i        (all agents share the mean reward)
```

Low ω prioritizes infection control; high ω prioritizes attendance. Sweeping ω traces the
safety–utility frontier.

## Project structure

- `environment/`
  - `multiclassroom.py` — `MultiClassroomEnv` (PettingZoo ParallelEnv). Supports a `risk_override`
    on `reset()` to evaluate on a fixed/real risk path while still randomizing initial infections.
  - `simulation.py` — infection dynamics (within-class + community-risk + cross-class coupling).
- `ppo_centralized.py` — Centralized PPO: a single Beta-policy controller over the full joint state.
- `ppo_ctde.py` — MAPPO-CTDE: decentralized Beta actors + a centralized critic.
- `analyze_environment.py` — diagnostic evaluation harness (matched information regimes + decompositions).
- `weekly_risk_sample_b.csv` — real weekly community-risk series (used by the "real" eval family).
- `EXPERIMENT_PLAN.docx` — research plan (RQs, hypotheses, experiments).

## Installation

```bash
conda create -n safecampus python=3.10 && conda activate safecampus
pip install -r requirements.txt
```

## Shared environment config

All three scripts must agree on the env for results to be comparable:
`TOTAL_STUDENTS=50`, `NUM_CLASSROOMS=2`, `MAX_WEEKS=15`, `cooperative_reward=True`,
`shared_fraction=0.3`, `OMEGA_VALUES=[0.1, …, 0.6]`.

## 1. Train the centralized controller

```bash
python ppo_centralized.py
```

- Single Beta-policy network over the full joint state (all rooms' infected + shared risk).
- Reward-only hyperparameter selection (learning rate, hidden dim) per ω.
- **Outputs:** training curve `centralized_ppo_results/training_rewards.png`; checkpoints
  `centralized_ppo_results/models/centralized_omega_{ω}_hd_{hidden_dim}_run_0.pt`.

## 2. Train the CTDE controller

```bash
python ppo_ctde.py
```

- Decentralized Beta actors (one per room, local obs only) with a shared centralized critic.
- `policy_type` can be `'beta'` (default), `'gaussian'`, or `'tanh'` (set in `main()`).
- **Outputs:** `mappo_results/combined_mappo_rewards_ci.png`,
  `mappo_results/combined_mappo_optimal_policies.png`; checkpoints
  `mappo_results/models/mappo_omega_{ω}_hd_{hidden_dim}_run_0.pt`.

### Training modes (both scripts, set in `main()`)
- `tune` — reward-only hyperparameter search.
- `train` — train with saved hyperparameters.
- `tune_and_train` — both (recommended for a first run).

## 3. Diagnostic evaluation

```bash
python analyze_environment.py
```

Compares controllers **within matched information regimes** rather than as a flat ranking:

| Regime | Methods |
|---|---|
| Full-info (full joint state) | DP (perfect-foresight ceiling), Joint-Myopic, **Centralized PPO** |
| Local-info (own room only) | Dec-Myopic, **CTDE MAPPO** |
| Floor | Random |

Evaluation runs over **K paired scenarios**, shared across methods, for two families:
- **synthetic** — held-out seeds → sampled risk + random initial infections (in-distribution);
- **real** — `weekly_risk_sample_b.csv` risk path fixed, initial infections varied over the seeds.

Reward-only metrics with bootstrap 95% CIs, plus named, paired **decompositions**: price of
decentralization (`Centralized − CTDE`), optimality gap (`DP − Centralized`), value of lookahead
(`Centralized − Joint-Myopic`). RL rows are skipped gracefully if no checkpoints exist yet.

**Outputs** (per family `{synthetic, real}`):
`rewards_by_league_{family}.png`, `normalized_score_{family}.png`, `decomposition_{family}.png`,
`reward_terms_{family}.png`, `trajectories_{family}.png`, plus `diagnostic_results.csv` and
`diagnostic_results.json`.

Key knobs at the top of `analyze_environment.py`: `K_SCENARIOS` (paired eval scenarios),
`DP_SCENARIOS` (DP subset, since DP is expensive), `N_ACTION_BINS` (DP/myopic action grid),
`OMEGA_VALUES`.

## Output layout

```
safecampus_multiclassroom/
├── centralized_ppo_results/
│   ├── models/centralized_omega_{ω}_hd_{hd}_run_0.pt
│   └── training_rewards.png
├── mappo_results/
│   ├── models/mappo_omega_{ω}_hd_{hd}_run_0.pt
│   ├── combined_mappo_rewards_ci.png
│   └── combined_mappo_optimal_policies.png
└── analysis_results/
    ├── rewards_by_league_{synthetic,real}.png
    ├── normalized_score_{synthetic,real}.png
    ├── decomposition_{synthetic,real}.png
    ├── reward_terms_{synthetic,real}.png
    ├── trajectories_{synthetic,real}.png
    ├── diagnostic_results.csv
    └── diagnostic_results.json
```

## Notes

- Actions are continuous capacity fractions in `[0, 1]`, scaled to `[0, TOTAL_STUDENTS]`.
- All training uses cooperative rewards (agents share a common objective).
- **Reward is the only performance metric** — hyperparameters are selected on reward alone.
- Centralized policy *structure* is intentionally not plotted: its action depends on the full
  `2·K`-dimensional state, so any static heatmap would require fixing the other rooms at arbitrary
  values. Behavior is instead read from on-distribution rollouts in the analysis (`trajectories_*`).
