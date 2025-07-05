from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from agents.myopic_agent import MyopicAgent
from utils.visualization import visualize_all_states_dqn,visualize_myopic_policy, visualize_all_states_centralized
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
import random
import torch
import optuna

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def run_experiment_centralized(
    learning_rate,
    hidden_dim,
    hidden_layers,
    total_students,
    num_classrooms,
    action_levels,
    gamma_values,
    alphas,
    method="mc",           # "mc" or "td"
    num_seeds=10,
    max_steps=73,
    out_dir="./results/centralized"
):
    """
    For each (gamma, alpha):
      1) Train once with train_centralized_mc or train_centralized_td.
      2) Visualize marginal policies via visualize_all_states_centralized.
      3) Evaluate across num_seeds (centralized eval).
      4) Emit three CSVs: global rewards, infected/allowed, training times.
    """
    np.random.seed(SEED)
    random.seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    global_rows = []
    ia_rows     = []
    time_rows   = []

    agents = [f"classroom_{i}" for i in range(num_classrooms)]

    for gamma in gamma_values:
        for alpha in alphas:
            # 1) instantiate env + agent
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

            agent = DQNAgent(
                agents,
                state_dim,
                act_n,
                reward_mix_alpha=alpha,
                gamma=gamma,
                seed=SEED,
                learning_rate=learning_rate,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                use_ctde=True
            )

            # 2) train
            t0 = time.time()
            if method == "mc":
                agent.train_centralized_mc(env, max_steps=max_steps)
            else:
                agent.train_centralized_td(env, max_steps=max_steps)
            train_time = time.time() - t0
            time_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "method": method,
                "train_time_s": train_time,
                "timestamp": timestamp
            })

            # 3) visualize
            vis_path = os.path.join(
                out_dir,
                f"centr_{method}_policy_γ{gamma:.2f}.png"
            )
            visualize_all_states_centralized(
                method,
                agents,
                env,
                agent,
                save_path=vis_path,
                gamma=gamma,
                grid_size=100
            )

            # 4) multi-seed evaluation
            rewards_accum  = []
            infected_accum = []
            allowed_accum  = []

            for _ in range(num_seeds):
                seed = random.randint(0, 2**31 - 1)
                env_eval = MultiClassroomEnv(
                    num_classrooms=num_classrooms,
                    total_students=total_students,
                    max_weeks=max_steps,
                    action_levels_per_class=[action_levels]*num_classrooms,
                    seed=seed,
                    gamma=gamma,
                    community_risk_data_file=None
                )
                # centralized eval
                data = agent.evaluate(env_eval, max_steps=max_steps, centralized=True)

                # global reward
                rewards_accum.append(
                    np.mean(list(data["total_rewards"].values()))
                )
                # avg infected & allowed
                infs = [np.mean(data["agents"][ag]["infected"]) for ag in agents]
                alls = [np.mean(data["agents"][ag]["allowed_students"]) for ag in agents]
                infected_accum.append(np.mean(infs))
                allowed_accum.append(np.mean(alls))

            global_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "method": method,
                "global_mean_reward": float(np.mean(rewards_accum)),
                "timestamp": timestamp
            })
            ia_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "method": method,
                "mean_infected": float(np.mean(infected_accum)),
                "mean_allowed": float(np.mean(allowed_accum)),
                "timestamp": timestamp
            })

    # 5) save summaries
    pd.DataFrame(global_rows).to_csv(
        os.path.join(out_dir, f"centr_{method}_global_summary.csv"), index=False
    )
    pd.DataFrame(ia_rows).to_csv(
        os.path.join(out_dir, f"centr_{method}_inf_allowed_summary.csv"), index=False
    )
    pd.DataFrame(time_rows).to_csv(
        os.path.join(out_dir, f"centr_{method}_time_summary.csv"), index=False
    )

    print(f"Centralized({method}) experiments saved under {out_dir}/")



def run_myopic_experiment(
    total_students,
    num_classrooms,
    action_levels,
    gamma_values,         # ← added
    alphas,
    num_seeds=10,
    max_steps=73,
    out_dir="./results/myopic"
):
    """
    For each (gamma, alpha):
      1) Instantiate myopic policy.
      2) Visualize policy map.
      3) Evaluate across num_seeds.
      4) Emit two CSVs with averaged metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    agents = [f"classroom_{i}" for i in range(num_classrooms)]

    global_rows = []
    ia_rows     = []

    for gamma in gamma_values:
        for alpha in alphas:
            # 1) Build myopic agent
            myopic = MyopicAgent(
                agents=agents,
                reward_mix_alpha=alpha,
                allowed_values=np.linspace(0, total_students, action_levels).astype(int)
            )

            # 2) Visualize learned policy
            #    Use a fresh env with a fixed seed so the color map is based on the true state‐space
            env_vis = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=total_students,
                max_weeks=max_steps,
                action_levels_per_class=[action_levels]*num_classrooms,
                seed=SEED,
                gamma=gamma,
                community_risk_data_file=None
            )
            env_vis.set_mode(True)
            vis_path = os.path.join(
                out_dir,
                f"myopic_policy_γ{gamma:.2f}_α{alpha:.2f}.png"
            )
            visualize_myopic_policy(
                agents,
                env_vis,
                myopic,
                save_path=vis_path,
                gamma=gamma,
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
                data = myopic.evaluate(env_eval, num_episodes=1, max_steps=max_steps)

                rewards_accum.append(np.mean(list(data["total_rewards"].values())))
                infs = [np.mean(data["agents"][ag]["infected"]) for ag in agents]
                alls = [np.mean(data["agents"][ag]["allowed_students"]) for ag in agents]
                infected_accum.append(np.mean(infs))
                allowed_accum.append(np.mean(alls))

            # 4) Aggregate and record
            global_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "global_mean_reward": float(np.mean(rewards_accum))
            })
            ia_rows.append({
                "gamma": gamma,
                "alpha": alpha,
                "mean_infected": float(np.mean(infected_accum)),
                "mean_allowed": float(np.mean(allowed_accum))
            })

    # 5) Save summaries
    pd.DataFrame(global_rows).to_csv(
        os.path.join(out_dir, "myopic_global_summary.csv"), index=False
    )
    pd.DataFrame(ia_rows).to_csv(
        os.path.join(out_dir, "myopic_inf_allowed_summary.csv"), index=False
    )

    print(f"Myopic summaries and visuals saved in {out_dir}/")

def run_dqn_experiment(
    learning_rate,
    hidden_dim,
    hidden_layers,
    method,
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
                             reward_mix_alpha=alpha, seed=SEED, learning_rate=learning_rate,
                             hidden_dim=hidden_dim, hidden_layers=hidden_layers)

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
            vis_path = os.path.join(out_dir, f"{method}_dqn_policy_{gamma}.png")
            visualize_all_states_dqn(
                method,
                agents,
                env,
                agent,
                gamma=gamma,
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
    pd.DataFrame(global_rows).to_csv(os.path.join(out_dir, f"{method}_dqn_global_summary.csv"), index=False)
    pd.DataFrame(ia_rows).    to_csv(os.path.join(out_dir, f"{method}_dqn_inf_allowed_summary.csv"), index=False)
    pd.DataFrame(time_rows).  to_csv(os.path.join(out_dir, f"{method}_dqn_time_summary.csv"), index=False)

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
    lr  = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
    hd  = trial.suggest_categorical("hidden_dim",   [32, 16])
    hl  = trial.suggest_int("hidden_layers", 1, 2)

    # fixed experiment settings
    total_students     = 100
    num_classrooms     = 2
    state_dim          = 2
    action_space_size  = 3
    action_levels      = 3
    gamma_values       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    eval_seeds         = [0,1,2]   # fewer seeds for speed
    max_steps          = 73
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
        agent.train_td(env_train, max_steps=max_steps)

        # evaluate over seeds
        for seed in eval_seeds:
            np.random.seed(seed)
            random.seed(seed)
            env_eval = make_env(seed, gamma, total_students, num_classrooms, max_steps, action_levels)
            data = agent.evaluate(env_eval, max_steps=max_steps)
            total_reward += sum(data["total_rewards"].values())

    # return the negative if we want to minimize; here we maximize
    return total_reward

def dqn_hyperparameter_tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, n_jobs=1, show_progress_bar=True)

    # get the best parameters
    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)

    # save to CSV
    df = pd.DataFrame([best_params])
    df.to_csv("exp5_centralized_td_hyperparameters.csv", index=False)
    print("Saved best hyperparams to best_dqn_hyperparameters.csv")

if __name__=="__main__":
    run_myopic_experiment(
        total_students=100,
        num_classrooms=2,
        action_levels=5,
        alphas=[1.0],
        gamma_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        num_seeds=10,
        max_steps=73,
        out_dir="./results/myopic"
    )

    # dqn_hyperparameter_tuning()



    run_dqn_experiment(
        learning_rate=0.00482,
        hidden_dim=16,
        hidden_layers=1,
        method="td",
        total_students=100,
        num_classrooms=2,
        action_levels=5,
        gamma_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        alphas=[1.0],
        num_seeds=10,
        max_steps=73,
        out_dir="./results"
    )

    run_dqn_experiment(
        learning_rate=0.002,
        hidden_dim=16,
        hidden_layers=2,
        method="mc",
        total_students=100,
        num_classrooms=2,
        action_levels=5,
        gamma_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        alphas=[1.0],
        num_seeds=10,
        max_steps=73,
        out_dir="./results"
    )
    # centralized Monte Carlo
    run_experiment_centralized(
        learning_rate=0.0272,
        hidden_dim=16,
        hidden_layers=1,
        total_students=100,
        num_classrooms=2,
        action_levels=5,
        gamma_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        alphas=[1.0],
        method="mc",
        num_seeds=10,
        max_steps=73,
        out_dir="./results/centralized_mc"
    )

    # centralized TD
    run_experiment_centralized(
        learning_rate=0.001,
        hidden_dim=32,
        hidden_layers=1,
        total_students=50,
        num_classrooms=2,
        action_levels=5,
        gamma_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        alphas=[1.0],
        method="td",
        num_seeds=10,
        max_steps=73,
        out_dir="./results/centralized_td"
    )

