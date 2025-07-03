from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import visualize_all_states_dqn
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
import random
import torch

SEED = 42

def run_dqn_experiment(
    total_students,
    num_classrooms,
    action_levels,
    gamma_values,
    alphas,
    num_seeds=10,
    max_steps=73,
    out_dir="./results"
):
    """
    For each (gamma, alpha):
      1) Train once.
      2) Visualize policy.
      3) Evaluate across num_seeds.
      4) Emit three CSVs with averaged metrics.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    global_rows = []
    ia_rows     = []
    time_rows   = []

    agents = [f"classroom_{i}" for i in range(num_classrooms)]
    for gamma in gamma_values:
        for alpha in alphas:
            # 1) Single training run
            env = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=total_students,
                max_weeks=max_steps,
                action_levels_per_class=[action_levels]*num_classrooms,
                seed=SEED,
                gamma=gamma,
                community_risk_data_file=None
            )
            state_dim = env.observation_spaces[agents[0]].shape[0]
            act_n     = env.action_spaces[agents[0]].n
            agent = DQNAgent(agents, state_dim, act_n,
                             reward_mix_alpha=alpha, gamma=gamma, seed=SEED)

            t0 = time.time()
            agent.train(env, max_steps=max_steps)
            train_time = time.time() - t0
            time_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "mean_training_time": train_time,
                "timestamp": timestamp
            })

            # 2) Visualize learned policy
            vis_path = os.path.join(out_dir, f"dqn_policy_γ{gamma}_α{alpha}_{timestamp}.png")
            visualize_all_states_dqn(
                agents,
                env,
                agent,
                save_path=vis_path,
                grid_size=100
            )

            # 3) Multi-seed evaluation
            rewards_accum   = []
            infected_accum  = []
            allowed_accum   = []

            for _ in range(num_seeds):
                seed = random.randint(0, 2**31-1)
                env_eval = MultiClassroomEnv(
                    num_classrooms=num_classrooms,
                    total_students=total_students,
                    max_weeks=max_steps,
                    action_levels_per_class=[action_levels]*num_classrooms,
                    seed=seed,
                    gamma=gamma,
                    community_risk_data_file=None
                )
                env_eval.set_mode(True)
                data = agent.evaluate(env_eval, max_steps=max_steps)

                # system-wide global reward
                rewards = list(data["total_rewards"].values())
                rewards_accum.append(np.mean(rewards))

                # avg infected & allowed per run
                infs = [np.mean(data["agents"][ag]["infected"]) for ag in agents]
                alls = [np.mean(data["agents"][ag]["allowed_students"]) for ag in agents]
                infected_accum.append(np.mean(infs))
                allowed_accum.append(np.mean(alls))

            # 4) Aggregate and record
            global_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "global_mean_reward": float(np.mean(rewards_accum)),
                "timestamp": timestamp
            })
            ia_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "mean_infected": float(np.mean(infected_accum)),
                "mean_allowed": float(np.mean(allowed_accum)),
                "timestamp": timestamp
            })

    # 5) Save summaries
    pd.DataFrame(global_rows).to_csv(os.path.join(out_dir, "dqn_global_summary.csv"), index=False)
    pd.DataFrame(ia_rows).    to_csv(os.path.join(out_dir, "dqn_inf_allowed_summary.csv"), index=False)
    pd.DataFrame(time_rows).  to_csv(os.path.join(out_dir, "dqn_time_summary.csv"), index=False)

    print(f"Saved summaries and visuals in {out_dir}/")

def make_env(seed, gamma, total_students, num_classrooms, max_steps, action_levels):
    env = MultiClassroomEnv(
        num_classrooms=num_classrooms,
        total_students=total_students,
        max_weeks=max_steps,
        action_levels_per_class=[action_levels]*num_classrooms,
        seed=seed,
        gamma=gamma,
        community_risk_data_file=None
    )
    return env

def objective(trial):
    # 1) sample hyperparameters
    lr  = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    hd  = trial.suggest_categorical("hidden_dim",   [16, 32, 64])
    hl  = trial.suggest_int("hidden_layers", 2, 4)

    # fixed experiment settings
    total_students     = 100
    num_classrooms     = 2
    state_dim          = 2
    action_space_size  = 3
    action_levels      = 3
    gamma_values       = [0.1, 0.2, 0.3]
    eval_seeds         = [0,1,2]   # fewer seeds for speed
    max_steps          = 52
    train_seed         = 0

    # aggregate reward across all gamma
    total_reward = 0.0
    for gamma in gamma_values:
        # train one agent
        agents = [f"classroom_{i}" for i in range(num_classrooms)]
        agent = DQNAgent(
            agents=agents,
            state_dim=state_dim,
            action_space_size=action_space_size,
            learning_rate=lr,
            hidden_dim=hd,
            hidden_layers=hl,
            gamma=gamma,
            seed=train_seed
        )
        env_train = make_env(train_seed, gamma, total_students, num_classrooms, max_steps, action_levels)
        agent.train(env_train, max_steps=max_steps)

        # evaluate over seeds
        for seed in eval_seeds:
            np.random.seed(seed)
            random.seed(seed)
            env_eval = make_env(seed, gamma, total_students, num_classrooms, max_steps, action_levels)
            data = agent.evaluate(env_eval, max_steps=max_steps)
            total_reward += sum(data["total_rewards"].values())

    # return the negative if we want to minimize; here we maximize
    return total_reward

if __name__=="__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # get the best parameters
    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)

    # save to CSV
    df = pd.DataFrame([best_params])
    df.to_csv("best_dqn_hyperparameters.csv", index=False)
    print("Saved best hyperparams to best_dqn_hyperparameters.csv")
