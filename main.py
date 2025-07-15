from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from agents.myopic_agent import MyopicAgent
from utils.visualization import visualize_all_states_dqn,visualize_myopic_policy, visualize_all_states_centralized
import numpy as np
import os
import psutil
import pandas as pd
import time
from datetime import datetime
import random
import torch
import optuna
from optuna.samplers import GridSampler

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

def run_ctde_experiment(
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
        alpha = 1.0
        env = MultiClassroomEnv(
            num_classrooms=num_classrooms,
            total_students=total_students,
            max_weeks=max_steps,
            action_levels_per_class=[action_levels] * num_classrooms,
            seed=SEED,
            gamma=gamma,
            community_risk_data_file=None
        )
        state_dim = env.observation_spaces[agents[0]].shape[0]
        act_n = env.action_spaces[agents[0]].n
        agent = DQNAgent(agents, state_dim, act_n,
                         reward_mix_alpha=alpha, seed=SEED, learning_rate=learning_rate,
                         hidden_dim=hidden_dim, hidden_layers=hidden_layers)

        t0 = time.time()
        agent.train_td_double(env, max_steps=max_steps, save_dir=out_dir)
        train_time = time.time() - t0
        time_rows.append({
            "gamma": gamma,
            "alpha": alpha,
            "mean_training_time": train_time,
            "timestamp": timestamp
        })

        # 2) Visualize learned policy
        vis_path = os.path.join(out_dir, f"{method}_ctde_policy_{gamma}.png")
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
        rewards_accum = []
        infected_accum = []
        allowed_accum = []

        for _ in range(num_seeds):
            seed = random.randint(0, 2 ** 31 - 1)
            env_eval = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=total_students,
                max_weeks=max_steps,
                action_levels_per_class=[action_levels] * num_classrooms,
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
    pd.DataFrame(global_rows).to_csv(os.path.join(out_dir, f"{method}_ctde_global_summary.csv"), index=False)
    pd.DataFrame(ia_rows).    to_csv(os.path.join(out_dir, f"{method}_ctde_inf_allowed_summary.csv"), index=False)
    pd.DataFrame(time_rows).  to_csv(os.path.join(out_dir, f"{method}_ctde_time_summary.csv"), index=False)

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

def objective_per_gamma(trial, gamma, num_eval_seeds=3):
    # 1) sample hyperparams
    lr = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1, 0.2])
    hd = trial.suggest_categorical("hidden_dim",      [128])
    hl = trial.suggest_categorical("hidden_layers",   [1, 2])

    total_students, num_classrooms = 100, 2
    max_steps, action_levels       = 73, 3

    agents = [f"classroom_{i}" for i in range(num_classrooms)]
    all_rewards = []

    for seed in range(num_eval_seeds):
        agent = DQNAgent(
            agents=agents,
            state_dim=2,
            action_space_size=3,
            learning_rate=lr,
            hidden_dim=hd,
            hidden_layers=hl,
            gamma=gamma,
            seed=seed
        )

        env_train = make_env(seed, gamma,
                             total_students, num_classrooms,
                             max_steps, action_levels)
        # assume this returns a list of per-episode *global* rewards
        rewards_hist = agent.train_td_hyper(
            env_train,
            max_steps=max_steps,
            save_dir=None,
            return_rewards=True
        )
        all_rewards.append(rewards_hist)

    # flatten across seeds: average reward per episode
    avg_rewards = np.mean(all_rewards, axis=0)

    # compute linear slope of that curve
    x = np.arange(len(avg_rewards))
    slope = np.polyfit(x, avg_rewards, 1)[0]

    # store it for later inspection
    trial.set_user_attr("reward_slope", float(slope))

    # if slope is negative, heavily penalize
    if slope < 0:
        return -1e6

    # otherwise return overall mean reward
    return float(np.mean(avg_rewards))


def dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1):
    results = []

    search_space = {
        "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 0.2],
        "hidden_dim":      [128],
        "hidden_layers":   [1]
    }

    for gamma in gamma_values:
        print(f"Tuning (reward + slope filter) for γ={gamma}")
        sampler = GridSampler(search_space)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        n_trials = (
            len(search_space["learning_rate"]) *
            len(search_space["hidden_dim"]) *
            len(search_space["hidden_layers"])
        )
        study.optimize(
            lambda t: objective_per_gamma(t, gamma, num_eval_seeds),
            n_trials=n_trials,
            show_progress_bar=True
        )

        # pick the best trial *among those with positive slope*
        valid = [t for t in study.trials if t.user_attrs.get("reward_slope", -999) > 0]
        if valid:
            best = max(valid, key=lambda t: t.value)
        else:
            # fallback if none had positive slope
            best = study.best_trial

        best_params = best.params
        best_params["gamma"] = gamma
        best_params["reward_slope"] = best.user_attrs.get("reward_slope")
        results.append(best_params)

        print(f"→ best for γ={gamma}: lr={best_params['learning_rate']}, "
              f"hd={best_params['hidden_dim']}, hl={best_params['hidden_layers']}, "
              f"slope={best_params['reward_slope']:.3f}")

    pd.DataFrame(results).to_csv(
        "best_dqn_hyperparameters_per_gamma.csv", index=False
    )
    print("Done: saved best configs (positive‐slope filtered) per gamma.")


# if __name__ == "__main__":
#     # GPU setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🎮 Using device: {device}")
#
#     if torch.cuda.is_available():
#         print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
#         print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
#         torch.cuda.empty_cache()
#     else:
#         print("⚠️  CUDA not available, using CPU")
#         # Set CPU affinity to cores 0-7 for DQN (only if using CPU)
#         try:
#             current_process = psutil.Process()
#             current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
#             print(f"🧠 DQN training set to use CPU cores 0-7")
#         except Exception as e:
#             print(f"⚠️  Could not set CPU affinity: {e}")
#
#     # Set process priority to be nice (lower priority)
#     os.nice(5)
#
#     # hyperparameter tuning
#     # gamma_values = [0.5]
#     # dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1)
#
#     # # --- 1) Manually enter best hyperparams per gamma ---
#     best_hyperparams = {
#         0.5: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 1},
#         # 0.7: {"learning_rate": 1e-2, "hidden_dim": 64, "hidden_layers": 8},
#         # 0.9: {"learning_rate": 1e-3, "hidden_dim": 128, "hidden_layers": 16},
#         # add more γ: {…} pairs as you like
#     }
#
#     # --- 2) Define your experimental sweep ---
#     total_students_list = [100]
#     num_classrooms_list = [2]
#     action_levels_list = [3]
#     alphas = [0.0]  # keep fixed or expand
#     num_seeds = 10
#     max_steps = 73
#
#     # --- 3) Loop over each γ with its own best hyperparams ---
#     for gamma, hyp in best_hyperparams.items():
#         lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]
#
#         for total_students in total_students_list:
#             for num_classrooms in num_classrooms_list:
#                 for action_levels in action_levels_list:
#                     out_dir = (
#                         f"./results/ctde/"
#                         f"γ{gamma}_ts{total_students}_nc{num_classrooms}_al{action_levels}"
#                     )
#                     os.makedirs(out_dir, exist_ok=True)
#
#                     print(
#                         f"CTDE | γ={gamma} | lr={lr} | hd={hd} | hl={hl} "
#                         f"| students={total_students} | classes={num_classrooms} "
#                         f"| levels={action_levels}"
#                     )
#
#                     run_ctde_experiment(
#                         learning_rate=lr,
#                         hidden_dim=hd,
#                         hidden_layers=hl,
#                         method="ctde",
#                         total_students=total_students,
#                         num_classrooms=num_classrooms,
#                         action_levels=action_levels,
#                         gamma_values=[gamma],  # only this γ
#                         alphas=alphas,
#                         num_seeds=num_seeds,
#                         max_steps=max_steps,
#                         out_dir=out_dir
#                     )
#
#     print("✅ All CTDE experiments with manual hyperparams are done.")

if __name__ == "__main__":
    # Set CPU affinity to cores 0-7 for DQN (leaving 8-15 for multiagent)
    try:
        current_process = psutil.Process()
        current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
        print(f"🧠 DQN training set to use CPU cores 0-7")
    except Exception as e:
        print(f"⚠️  Could not set CPU affinity: {e}")

    # Set process priority to be nice (lower priority)
    os.nice(5)

    # hyperparameter tuning
    # gamma_values = [0.5]
    # dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1)

    # # --- 1) Manually enter best hyperparams per gamma ---
    best_hyperparams = {
        0.5: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 1},
        # 0.7: {"learning_rate": 1e-2, "hidden_dim": 64, "hidden_layers": 8},
        # 0.9: {"learning_rate": 1e-3, "hidden_dim": 128, "hidden_layers": 16},
        # add more γ: {…} pairs as you like
    }

    # --- 2) Define your experimental sweep ---
    total_students_list = [100]
    num_classrooms_list = [2]
    action_levels_list = [3]
    alphas = [0.0]  # keep fixed or expand
    num_seeds = 10
    max_steps = 73

    # --- 3) Loop over each γ with its own best hyperparams ---
    for gamma, hyp in best_hyperparams.items():
        lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]

        for total_students in total_students_list:
            for num_classrooms in num_classrooms_list:
                for action_levels in action_levels_list:
                    out_dir = (
                        f"./results/ctde/"
                        f"γ{gamma}_ts{total_students}_nc{num_classrooms}_al{action_levels}"
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    print(
                        f"CTDE | γ={gamma} | lr={lr} | hd={hd} | hl={hl} "
                        f"| students={total_students} | classes={num_classrooms} "
                        f"| levels={action_levels}"
                    )

                    run_ctde_experiment(
                        learning_rate=lr,
                        hidden_dim=hd,
                        hidden_layers=hl,
                        method="ctde",
                        total_students=total_students,
                        num_classrooms=num_classrooms,
                        action_levels=action_levels,
                        gamma_values=[gamma],  # only this γ
                        alphas=alphas,
                        num_seeds=num_seeds,
                        max_steps=max_steps,
                        out_dir=out_dir
                    )

    print("✅ All CTDE experiments with manual hyperparams are done.")

