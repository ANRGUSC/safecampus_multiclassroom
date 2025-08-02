from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from agents.myopic_agent import MyopicAgent
from utils.visualization import visualize_all_states_dqn,visualize_myopic_policy, visualize_all_states_centralized, \
    visualize_all_states_dqn_swap
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
random.seed(SEED)

def run_experiment_centralized(
    learning_rate,
    hidden_dim,
    hidden_layers,
    total_students,
    num_classrooms,
    action_levels,
    gamma_values,
    alphas,
    method="mc",  # "mc" or "td"
    num_seeds=10,
    max_steps=73,
    out_dir="./results/centralized_mc",
    global_rows=None,
    ia_rows=None,
    time_rows=None,
):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    if global_rows is None:
        global_rows = []
    if ia_rows is None:
        ia_rows = []
    if time_rows is None:
        time_rows = []

    agents = [f"classroom_{i}" for i in range(num_classrooms)]

    for gamma in gamma_values:
        for alpha in alphas:
            env = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=total_students,
                max_weeks=max_steps,
                action_levels_per_class=[action_levels] * num_classrooms,
                seed=SEED,
                gamma=gamma,
                community_risk_data_file=None,
            )
            state_dim = env.observation_spaces[agents[0]].shape[0]
            act_n = env.action_spaces[agents[0]].n

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
                use_ctde=True,
            )

            t0 = time.time()
            if method == "mc":
                agent.train_centralized_mc(env, max_steps=max_steps, save_dir=out_dir)
            else:
                agent.train_centralized_td_double(env, max_steps=max_steps, save_dir=out_dir)
            train_time = time.time() - t0

            time_rows.append(
                {
                    "gamma": gamma,
                    "alpha": alpha,
                    "method": method,
                    "train_time_s": train_time,
                    "timestamp": timestamp,
                }
            )

            # Visualize learned policy once
            vis_path = os.path.join(out_dir, f"{method}_centr_policy_{gamma}.png")
            visualize_all_states_centralized(
                method, agents, env, agent, save_path=vis_path, gamma=gamma, grid_size=100
            )

            # Accumulators for normal evaluation
            rewards_accum = []
            infected_accum = []
            allowed_accum = []

            # Accumulators for single-policy-for-all evaluation (per policy agent scalar rewards)
            single_policy_scalar_rewards_accum = {pa: [] for pa in agents}

            for _ in range(num_seeds):
                seed = random.randint(0, 2**31 - 1)
                env_eval = MultiClassroomEnv(
                    num_classrooms=num_classrooms,
                    total_students=total_students,
                    max_weeks=max_steps,
                    action_levels_per_class=[action_levels] * num_classrooms,
                    seed=seed,
                    gamma=gamma,
                    community_risk_data_file=None,
                )
                env_eval.set_mode(True)

                # Normal evaluation
                data_normal = agent.evaluate(
                    env_eval,
                    max_steps=max_steps,
                    centralized=True,
                    evaluate_cross_policy=False,
                    evaluate_single_policy_all=False,
                )
                rewards = list(data_normal["total_rewards"].values())
                rewards_accum.append(np.mean(rewards))

                infs = [np.mean(data_normal["agents"][ag]["infected"]) for ag in agents]
                alls = [np.mean(data_normal["agents"][ag]["allowed_students"]) for ag in agents]
                infected_accum.append(np.mean(infs))
                allowed_accum.append(np.mean(alls))

                # Single policy for all evaluation
                data_single_policy = agent.evaluate(
                    env_eval,
                    max_steps=max_steps,
                    centralized=True,
                    evaluate_cross_policy=False,
                    evaluate_single_policy_all=True,
                )

                # Aggregate scalar rewards for single-policy-for-all (mean over classrooms)
                for policy_agent, rewards_dict in data_single_policy["policy_rewards"].items():
                    mean_reward = np.mean(list(rewards_dict.values()))
                    single_policy_scalar_rewards_accum[policy_agent].append(mean_reward)

            # Save normal evaluation summary
            global_rows.append(
                {
                    "gamma": gamma,
                    "alpha": alpha,
                    "method": method,
                    "global_mean_reward": float(np.mean(rewards_accum)),
                    "timestamp": timestamp,
                }
            )
            ia_rows.append(
                {
                    "gamma": gamma,
                    "alpha": alpha,
                    "method": method,
                    "mean_infected": float(np.mean(infected_accum)),
                    "mean_allowed": float(np.mean(allowed_accum)),
                    "timestamp": timestamp,
                }
            )

            # Compute mean rewards per policy agent averaged over seeds
            policy_mean_rewards = {
                pa: np.mean(rewards)
                for pa, rewards in single_policy_scalar_rewards_accum.items()
            }

            # Overall mean of single policy rewards (averaged across policies)
            overall_single_policy_mean = np.mean(list(policy_mean_rewards.values()))

            global_rows.append(
                {
                    "gamma": gamma,
                    "alpha": alpha,
                    "method": method,
                    "global_mean_single_policy_reward": overall_single_policy_mean,
                    "timestamp": timestamp,
                }
            )

            # Save per policy agent mean rewards for detailed analysis
            for policy_agent, mean_reward in policy_mean_rewards.items():
                ia_rows.append(
                    {
                        "gamma": gamma,
                        "alpha": alpha,
                        "method": method,
                        "policy_agent": policy_agent,
                        "mean_reward": mean_reward,
                        "timestamp": timestamp,
                    }
                )

    return global_rows, ia_rows, time_rows


def run_myopic_experiment(
    total_students,
    num_classrooms,
    action_levels,
    gamma_values,
    alphas,
    num_seeds=10,
    max_steps=73,
    out_dir="./results/myopic"
):
    """
    For each (gamma, alpha):
      1) Instantiate myopic agent.
      2) Visualize policy.
      3) Evaluate across seeds.
      4) Save averaged metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    agents = [f"classroom_{i}" for i in range(num_classrooms)]

    global_rows = []
    ia_rows = []

    for gamma in gamma_values:
        for alpha in alphas:
            # Instantiate myopic agent
            myopic = MyopicAgent(
                agents=agents,
                reward_mix_alpha=alpha,
                allowed_values=np.linspace(0, total_students, action_levels).astype(int)
            )

            # Visualization environment setup
            env_vis = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=total_students,
                max_weeks=max_steps,
                action_levels_per_class=[action_levels] * num_classrooms,
                seed=SEED,
                gamma=gamma,
                community_risk_data_file=None
            )
            env_vis.set_mode(True)
            vis_path = os.path.join(out_dir, f"myopic_policy_γ{gamma:.2f}_shared0.6.png")
            visualize_myopic_policy(
                agents,
                env_vis,
                myopic,
                save_path=vis_path,
                gamma=gamma,
                grid_size=100
            )

            # Accumulators
            rewards_accum = []
            infected_accum = []
            allowed_accum = []

            for _ in range(num_seeds):
                seed = random.randint(0, 2**31 - 1)
                env_eval = MultiClassroomEnv(
                    num_classrooms=num_classrooms,
                    total_students=total_students,
                    max_weeks=max_steps,
                    action_levels_per_class=[action_levels] * num_classrooms,
                    seed=seed,
                    gamma=gamma,
                    community_risk_data_file=None,
                )
                env_eval.set_mode(True)
                data = myopic.evaluate(env_eval, num_episodes=1, max_steps=max_steps)

                rewards_accum.append(np.mean(list(data["total_rewards"].values())))
                infs = [np.mean(data["agents"][ag]["infected"]) for ag in agents]
                alls = [np.mean(data["agents"][ag]["allowed_students"]) for ag in agents]
                infected_accum.append(np.mean(infs))
                allowed_accum.append(np.mean(alls))

            # Aggregate results
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

    # Save summaries
    pd.DataFrame(global_rows).to_csv(os.path.join(out_dir, "myopic_global_summary.csv"), index=False)
    pd.DataFrame(ia_rows).to_csv(os.path.join(out_dir, "myopic_inf_allowed_summary.csv"), index=False)

    print(f"Myopic summaries and visuals saved in {out_dir}/")

    return global_rows, ia_rows



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
    out_dir="./results",
    global_rows=None,
    ia_rows=None,
    time_rows=None,
):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    if global_rows is None:
        global_rows = []
    if ia_rows is None:
        ia_rows = []
    if time_rows is None:
        time_rows = []

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
            community_risk_data_file=None,
        )
        state_dim = env.observation_spaces[agents[0]].shape[0]
        act_n = env.action_spaces[agents[0]].n
        agent = DQNAgent(
            agents,
            state_dim,
            act_n,
            reward_mix_alpha=alpha,
            seed=SEED,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )

        t0 = time.time()
        if method == "mc":
            agent.train_mc(env, max_steps=max_steps, save_dir=out_dir)
        else:
            agent.train_td_double(env, max_steps=max_steps, save_dir=out_dir)
        train_time = time.time() - t0
        time_rows.append(
            {
                "gamma": gamma,
                "alpha": alpha,
                "mean_training_time": train_time,
                "timestamp": timestamp,
            }
        )

        # Visualize learned policy once
        vis_path = os.path.join(out_dir, f"{method}_ctde_policy_{gamma}.png")
        visualize_all_states_dqn(
            method, agents, env, agent, gamma=gamma, save_path=vis_path, grid_size=100
        )

        for swap_agent in agents:
            swap_vis_path = os.path.join(
                out_dir, f"{method}_ctde_policy_swap_{swap_agent}_{gamma}.png"
            )
            visualize_all_states_dqn_swap(
                method,
                agents,
                env,
                agent,
                save_path=swap_vis_path,
                gamma=gamma,
                grid_size=100,
                swap_policy_agent=swap_agent
            )

        # Accumulators for normal evaluation
        rewards_accum = []
        infected_accum = []
        allowed_accum = []

        # Accumulators for single-policy-for-all eval
        single_policy_scalar_rewards_accum = {pa: [] for pa in agents}

        for _ in range(num_seeds):
            seed = random.randint(0, 2**31 - 1)
            env_eval = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=total_students,
                max_weeks=max_steps,
                action_levels_per_class=[action_levels] * num_classrooms,
                seed=seed,
                gamma=gamma,
                community_risk_data_file=None,
            )
            env_eval.set_mode(True)

            # Normal evaluation
            data_normal = agent.evaluate(
                env_eval,
                max_steps=max_steps,
                centralized=False,
                evaluate_cross_policy=False,
                evaluate_single_policy_all=False,
            )
            rewards = list(data_normal["total_rewards"].values())
            rewards_accum.append(np.mean(rewards))

            infs = [np.mean(data_normal["agents"][ag]["infected"]) for ag in agents]
            alls = [np.mean(data_normal["agents"][ag]["allowed_students"]) for ag in agents]
            infected_accum.append(np.mean(infs))
            allowed_accum.append(np.mean(alls))

            # Single policy for all evaluation
            data_single_policy = agent.evaluate(
                env_eval,
                max_steps=max_steps,
                centralized=False,
                evaluate_cross_policy=False,
                evaluate_single_policy_all=True,
            )

            # Aggregate scalar rewards for single-policy-for-all
            for policy_agent, rewards_dict in data_single_policy["policy_rewards"].items():
                mean_reward = np.mean(list(rewards_dict.values()))  # Average across classrooms
                single_policy_scalar_rewards_accum[policy_agent].append(mean_reward)

        # Save normal evaluation summary
        global_rows.append(
            {
                "gamma": gamma,
                "alpha": alpha,
                "global_mean_reward": float(np.mean(rewards_accum)),
                "timestamp": timestamp,
            }
        )
        ia_rows.append(
            {
                "gamma": gamma,
                "alpha": alpha,
                "mean_infected": float(np.mean(infected_accum)),
                "mean_allowed": float(np.mean(allowed_accum)),
                "timestamp": timestamp,
            }
        )

        # Compute mean rewards per policy agent averaged over seeds
        policy_mean_rewards = {
            pa: np.mean(rewards)
            for pa, rewards in single_policy_scalar_rewards_accum.items()
        }

        # Overall mean of single policy rewards (averaged across policies)
        overall_single_policy_mean = np.mean(list(policy_mean_rewards.values()))

        global_rows.append(
            {
                "gamma": gamma,
                "alpha": alpha,
                "global_mean_single_policy_reward": overall_single_policy_mean,
                "timestamp": timestamp,
            }
        )

        # Save per policy and classroom mean rewards for detailed analysis
        for policy_agent, rewards in policy_mean_rewards.items():
            # Since single-policy-for-all gives mean across classrooms, we log per policy agent only
            ia_rows.append(
                {
                    "gamma": gamma,
                    "alpha": alpha,
                    "policy_agent": policy_agent,
                    "mean_reward": rewards,
                    "timestamp": timestamp,
                }
            )

    return global_rows, ia_rows, time_rows


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

# Tuning with loss
def objective_per_gamma(trial, gamma, num_eval_seeds=3, use_loss=False):
    # Sample hyperparameters
    lr = trial.suggest_categorical("learning_rate", [0.0001, 0.001])
    hd = trial.suggest_categorical("hidden_dim", [64, 128])
    hl = trial.suggest_categorical("hidden_layers", [1, 2, 3])

    total_students, num_classrooms = 100, 2
    max_steps, action_levels = 73, 3

    agents = [f"classroom_{i}" for i in range(num_classrooms)]
    all_vals = []

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

        if use_loss:
            # Get per-episode loss history
            loss_hist = agent.train_centralized_td_double_hyper_loss(
                env_train,
                max_steps=max_steps,
                return_loss=True
            )
            all_vals.append(loss_hist)
        else:
            # Get per-episode reward history
            reward_hist = agent.train_centralized_td_double_hyper_rewards(
                env_train,
                max_steps=max_steps,
                return_rewards=True
            )
            all_vals.append(reward_hist)

    avg_curve = np.mean(all_vals, axis=0)
    mean_val = float(np.mean(avg_curve))

    if use_loss:
        trial.set_user_attr("mean_loss", mean_val)
        return mean_val  # minimize
    else:
        trial.set_user_attr("mean_reward", mean_val)
        return mean_val  # maximize


def dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1, use_loss=False):
    results = []

    search_space = {
        "learning_rate": [0.0001, 0.001],
        "hidden_dim": [64, 128],
        "hidden_layers": [1, 2, 3]
    }

    direction = "minimize" if use_loss else "maximize"

    for gamma in gamma_values:
        print(f"Tuning ({'loss' if use_loss else 'reward'}) for γ={gamma}")
        sampler = GridSampler(search_space)
        study = optuna.create_study(direction=direction, sampler=sampler)

        n_trials = (
            len(search_space["learning_rate"]) *
            len(search_space["hidden_dim"]) *
            len(search_space["hidden_layers"])
        )

        study.optimize(
            lambda t: objective_per_gamma(t, gamma, num_eval_seeds, use_loss),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best = study.best_trial
        best_params = best.params
        best_params["gamma"] = gamma

        if use_loss:
            best_params["mean_loss"] = best.user_attrs.get("mean_loss")
            val_str = (
                f"{best_params['mean_loss']:.3f}" if best_params["mean_loss"] is not None else "N/A"
            )
            print(f"→ best for γ={gamma}: lr={best_params['learning_rate']}, "
                  f"hd={best_params['hidden_dim']}, hl={best_params['hidden_layers']}, "
                  f"mean_loss={val_str}")
        else:
            best_params["mean_reward"] = best.user_attrs.get("mean_reward")
            val_str = (
                f"{best_params['mean_reward']:.3f}" if best_params["mean_reward"] is not None else "N/A"
            )
            print(f"→ best for γ={gamma}: lr={best_params['learning_rate']}, "
                  f"hd={best_params['hidden_dim']}, hl={best_params['hidden_layers']}, "
                  f"mean_reward={val_str}")

        results.append(best_params)

    filename = "best_ctde_td_hyperparameters_per_gamma.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Done: saved best configs ({'loss' if use_loss else 'reward'}) per gamma to {filename}")




# if __name__ == "__main__":
#     try:
#         current_process = psutil.Process()
#         current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
#         print(f"🧠 DQN training set to use CPU cores 0-7")
#     except Exception as e:
#         print(f"⚠️ Could not set CPU affinity: {e}")
#
#     os.nice(5)
#     # hyperparameter tuning
#     # gamma_values = [0.3, 0.4, 0.5]
#     # dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1, use_loss=True)
#
#     best_hyperparams = {
#         # 0.1: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
#         # 0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
#         # 0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
#         0.4: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
#         # 0.5: {"learning_rate": 0.1, "hidden_dim": 128, "hidden_layers": 3},
#         # 0.6: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
#     }
#
#     total_students_list = [100]
#     num_classrooms_list = [2]
#     action_levels_list = [3]
#     alphas = [0.0]
#     num_seeds = 10
#     max_steps = 73
#     gamma_values = list(best_hyperparams.keys())
#
#     for total_students in total_students_list:
#         for num_classrooms in num_classrooms_list:
#             for action_levels in action_levels_list:
#                 out_dir = (
#                     f"./results/ctde-td/"
#                     f"all_gamma_ts{total_students}_nc{num_classrooms}_al{action_levels}_sf{0.6}"
#                 )
#                 os.makedirs(out_dir, exist_ok=True)
#
#                 print(f"📦 Running CTDE-TD sweep over gamma in one folder: {out_dir}")
#                 global_rows = []
#                 ia_rows = []
#                 time_rows = []
#
#                 for gamma in gamma_values:
#                     hyp = best_hyperparams[gamma]
#                     lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]
#
#                     print(
#                         f"→ CTDE-TD | γ={gamma} | lr={lr} | hd={hd} | hl={hl} "
#                         f"| students={total_students} | classes={num_classrooms} "
#                         f"| levels={action_levels}"
#                     )
#
#                     global_rows, ia_rows, time_rows = run_ctde_experiment(
#                         learning_rate=lr,
#                         hidden_dim=hd,
#                         hidden_layers=hl,
#                         method="td",
#                         total_students=total_students,
#                         num_classrooms=num_classrooms,
#                         action_levels=action_levels,
#                         gamma_values=[gamma],
#                         alphas=alphas,
#                         num_seeds=num_seeds,
#                         max_steps=max_steps,
#                         out_dir=out_dir,
#                         global_rows=global_rows,
#                         ia_rows=ia_rows,
#                         time_rows=time_rows,
#                     )
#
#                 # Separate normal and cross-policy results for saving
#                 normal_global = [row for row in global_rows if "global_mean_reward" in row]
#                 normal_ia = [row for row in ia_rows if "mean_infected" in row and "permutation" not in row and "policy_agent" not in row]
#
#                 # cross_global = [row for row in global_rows if "global_mean_cross_policy_reward" in row]
#                 # cross_ia = [row for row in ia_rows if "permutation" in row]
#
#                 single_global = [row for row in global_rows if "global_mean_single_policy_reward" in row]
#                 single_ia = [row for row in ia_rows if "policy_agent" in row]
#
#                 # Save normal evaluation CSVs
#                 pd.DataFrame(normal_global).to_csv(os.path.join(out_dir, "ctde_td_global_summary.csv"), index=False)
#                 pd.DataFrame(normal_ia).to_csv(os.path.join(out_dir, "ctde_td_inf_allowed_summary.csv"), index=False)
#
#                 # Save cross-policy evaluation CSVs
#                 # pd.DataFrame(cross_global).to_csv(os.path.join(out_dir, "ctde_mc_cross_policy_global_summary.csv"), index=False)
#                 # pd.DataFrame(cross_ia).to_csv(os.path.join(out_dir, "ctde_mc_cross_policy_perm_summary.csv"), index=False)
#
#                 # Save single-policy-for-all evaluation CSVs
#                 # pd.DataFrame(single_global).to_csv(os.path.join(out_dir, "ctde_td_single_policy_global_summary.csv"), index=False)
#                 # pd.DataFrame(single_ia).to_csv(os.path.join(out_dir, "ctde_td_single_policy_classroom_summary.csv"), index=False)
#
#                 # Save time summary CSV
#                 pd.DataFrame(time_rows).to_csv(os.path.join(out_dir, "ctde_td_time_summary.csv"), index=False)
#
#     print("✅ All CTDE experiments with manual hyperparams are done.")


if __name__ == "__main__":
    try:
        current_process = psutil.Process()
        current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
        print(f"🧠 DQN training set to use CPU cores 0-7")
    except Exception as e:
        print(f"⚠️ Could not set CPU affinity: {e}")

    os.nice(5)
    # gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1, use_loss=True)

    best_hyperparams = {
        # 0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
        # 0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
        # 0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
        0.4: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
        # 0.5: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
        # 0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
    }

    total_students_list = [100]
    num_classrooms_list = [2]
    action_levels_list = [3]
    alphas = [0.0]
    num_seeds = 10
    max_steps = 73
    gamma_values = list(best_hyperparams.keys())

    for total_students in total_students_list:
        for num_classrooms in num_classrooms_list:
            for action_levels in action_levels_list:
                out_dir = (
                    f"./results/centralized_mc/"
                    f"all_gamma_ts{total_students}_nc{num_classrooms}_al{action_levels}_sf{0.6}"
                )
                os.makedirs(out_dir, exist_ok=True)

                print(f"📦 Running Centralized MC sweep over gamma in one folder: {out_dir}")
                global_rows = []
                ia_rows = []
                time_rows = []

                for gamma in gamma_values:
                    hyp = best_hyperparams[gamma]
                    lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]

                    print(
                        f"→ Centralized-MC | γ={gamma} | lr={lr} | hd={hd} | hl={hl} "
                        f"| students={total_students} | classes={num_classrooms} "
                        f"| levels={action_levels}"
                    )

                    global_rows, ia_rows, time_rows = run_experiment_centralized(
                        learning_rate=lr,
                        hidden_dim=hd,
                        hidden_layers=hl,
                        method="td",
                        total_students=total_students,
                        num_classrooms=num_classrooms,
                        action_levels=action_levels,
                        gamma_values=[gamma],
                        alphas=alphas,
                        num_seeds=num_seeds,
                        max_steps=max_steps,
                        out_dir=out_dir,
                        global_rows=global_rows,
                        ia_rows=ia_rows,
                        time_rows=time_rows,
                    )

                # Separate normal and single-policy results for saving
                normal_global = [row for row in global_rows if "global_mean_reward" in row]
                normal_ia = [row for row in ia_rows if "mean_infected" in row and "policy_agent" not in row]

                single_global = [row for row in global_rows if "global_mean_single_policy_reward" in row]
                single_ia = [row for row in ia_rows if "policy_agent" in row]

                # Save normal evaluation CSVs
                pd.DataFrame(normal_global).to_csv(os.path.join(out_dir, "centralized_td_global_summary.csv"), index=False)
                pd.DataFrame(normal_ia).to_csv(os.path.join(out_dir, "centralized_td_inf_allowed_summary.csv"), index=False)

                # Save single-policy-for-all evaluation CSVs
                pd.DataFrame(single_global).to_csv(os.path.join(out_dir, "centralized_td_single_policy_global_summary.csv"), index=False)
                pd.DataFrame(single_ia).to_csv(os.path.join(out_dir, "centralized_td_single_policy_classroom_summary.csv"), index=False)

                # Save time summary CSV
                pd.DataFrame(time_rows).to_csv(os.path.join(out_dir, "centralized_td_time_summary.csv"), index=False)

    print("✅ All Centralized MC experiments with manual hyperparams are done.")



# if __name__ == "__main__":
#
#     try:
#         current_process = psutil.Process()
#         current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
#         print(f"🧠 DQN Centralized training set to use CPU cores 0-7")
#     except Exception as e:
#         print(f"⚠️ Could not set CPU affinity: {e}")
#
#     os.nice(5)
#
#     best_hyperparams = {
#         0.1: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 1},
#         0.2: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
#         0.3: {"learning_rate": 0.03, "hidden_dim": 64, "hidden_layers": 1},
#         0.4: {"learning_rate": 0.01, "hidden_dim": 128, "hidden_layers": 2},
#         0.5: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 2},
#         0.6: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 2},
#     }
#
#     total_students_list = [100]
#     num_classrooms_list = [2]
#     action_levels_list = [3]
#     alphas = [0.0]
#     num_seeds = 10
#     max_steps = 73
#     gamma_values = list(best_hyperparams.keys())
#
#     for total_students in total_students_list:
#         for num_classrooms in num_classrooms_list:
#             for action_levels in action_levels_list:
#                 out_dir = (
#                     f"./results/centralized_mc/"
#                     f"all_gamma_ts{total_students}_nc{num_classrooms}_al{action_levels}"
#                 )
#                 os.makedirs(out_dir, exist_ok=True)
#
#                 print(f"📦 Running centralized sweep over gamma in folder: {out_dir}")
#                 global_rows = []
#                 ia_rows = []
#                 time_rows = []
#
#                 for gamma in gamma_values:
#                     hyp = best_hyperparams[gamma]
#                     lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]
#
#                     print(
#                         f"→ CENTRALIZED-MC | γ={gamma} | lr={lr} | hd={hd} | hl={hl} "
#                         f"| students={total_students} | classes={num_classrooms} "
#                         f"| levels={action_levels}"
#                     )
#
#                     global_rows, ia_rows, time_rows = run_experiment_centralized(
#                         learning_rate=lr,
#                         hidden_dim=hd,
#                         hidden_layers=hl,
#                         total_students=total_students,
#                         num_classrooms=num_classrooms,
#                         action_levels=action_levels,
#                         gamma_values=[gamma],
#                         alphas=alphas,
#                         method="mc",
#                         num_seeds=num_seeds,
#                         max_steps=max_steps,
#                         out_dir=out_dir,
#                         global_rows=global_rows,
#                         ia_rows=ia_rows,
#                         time_rows=time_rows,
#                     )
#
#                 # Separate and save normal evaluation results
#                 normal_global = [row for row in global_rows if "global_mean_reward" in row]
#                 normal_ia = [row for row in ia_rows if "mean_infected" in row and "policy_agent" not in row and "permutation" not in row]
#
#                 # Separate and save cross-policy evaluation results
#                 cross_global = [row for row in global_rows if "global_mean_cross_policy_reward" in row]
#                 cross_ia = [row for row in ia_rows if "permutation" in row]
#
#                 # Separate and save single-policy-all evaluation results
#                 single_global = [row for row in global_rows if "global_mean_single_policy_reward" in row]
#                 single_ia = [row for row in ia_rows if "policy_agent" in row]
#
#                 pd.DataFrame(normal_global).to_csv(os.path.join(out_dir, "centralized_mc_global_summary.csv"), index=False)
#                 pd.DataFrame(normal_ia).to_csv(os.path.join(out_dir, "centralized_mc_inf_allowed_summary.csv"), index=False)
#
#                 pd.DataFrame(cross_global).to_csv(os.path.join(out_dir, "centralized_mc_cross_policy_global_summary.csv"), index=False)
#                 pd.DataFrame(cross_ia).to_csv(os.path.join(out_dir, "centralized_mc_cross_policy_perm_summary.csv"), index=False)
#
#                 pd.DataFrame(single_global).to_csv(os.path.join(out_dir, "centralized_mc_single_policy_global_summary.csv"), index=False)
#                 pd.DataFrame(single_ia).to_csv(os.path.join(out_dir, "centralized_mc_single_policy_agent_summary.csv"), index=False)
#
#                 pd.DataFrame(time_rows).to_csv(os.path.join(out_dir, "centralized_mc_time_summary.csv"), index=False)
#
#     print("✅ All centralized experiments with manual hyperparams are done.")










# if __name__ == "__main__":
#     # Set CPU affinity to cores 0-7 for DQN (leaving 8-15 for multiagent)
#     try:
#         current_process = psutil.Process()
#         current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
#         print(f"🧠 DQN training set to use CPU cores 0-7")
#     except Exception as e:
#         print(f"⚠️  Could not set CPU affinity: {e}")
#
#     # Set process priority to be nice (lower priority)
#     os.nice(5)
#
#     # hyperparameter tuning
#     # gamma_values = [0.1, 0.2, 0.3, 0.4,0.5, 0.6]
#     # dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1, use_loss=True)
#
#     # --- 1) Manually enter best hyperparams per gamma ---
#     best_hyperparams = {
#         0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
#         0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
#         0.3: {"learning_rate": 0.00001, "hidden_dim": 256, "hidden_layers": 4},
#         0.4: {"learning_rate": 0.000001, "hidden_dim": 256, "hidden_layers": 4},
#         0.5: {"learning_rate": 0.000001, "hidden_dim": 256, "hidden_layers": 4},
#         0.6: {"learning_rate": 0.001, "hidden_dim":64, "hidden_layers": 3},
#     }
#
#     # --- 2) Define your experimental sweep ---
#     total_students_list = [100]
#     num_classrooms_list = [3]
#     action_levels_list = [3]
#     alphas = [0.0]  # keep fixed or expand
#     num_seeds = 10
#     max_steps = 73
#     gamma_values = list(best_hyperparams.keys())
#     # Shared logs
#
#     # --- 3) Run all CTDE experiments in one unified directory ---
#     for total_students in total_students_list:
#         for num_classrooms in num_classrooms_list:
#             for action_levels in action_levels_list:
#                 # Shared output dir for all gammas
#                 out_dir = (
#                     f"./results/ctde-td/"
#                     f"all_gamma_ts{total_students}_nc{num_classrooms}_al{action_levels}"
#                 )
#                 os.makedirs(out_dir, exist_ok=True)
#
#                 print(f"📦 Running CTDE-TD sweep over gamma in one folder: {out_dir}")
#                 global_rows = []
#                 ia_rows = []
#                 time_rows = []
#
#                 # Run all gamma values using their respective hyperparameters
#                 for gamma in gamma_values:
#                     hyp = best_hyperparams[gamma]
#                     lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]
#
#                     print(
#                         f"→ CTDE-TD | γ={gamma} | lr={lr} | hd={hd} | hl={hl} "
#                         f"| students={total_students} | classes={num_classrooms} "
#                         f"| levels={action_levels}"
#                     )
#
#                     global_rows, ia_rows, time_rows = run_ctde_experiment(
#                         learning_rate=lr,
#                         hidden_dim=hd,
#                         hidden_layers=hl,
#                         method="td",
#                         total_students=total_students,
#                         num_classrooms=num_classrooms,
#                         action_levels=action_levels,
#                         gamma_values=[gamma],  # run 1 gamma at a time
#                         alphas=alphas,
#                         num_seeds=num_seeds,
#                         max_steps=max_steps,
#                         out_dir=out_dir,
#                         global_rows=global_rows,
#                         ia_rows=ia_rows,
#                         time_rows=time_rows
#                     )
#
#                 # Save once after all gammas
#                 pd.DataFrame(global_rows).to_csv(os.path.join(out_dir, "ctde_td_global_summary.csv"), index=False)
#                 pd.DataFrame(ia_rows).to_csv(os.path.join(out_dir, "ctde_td_inf_allowed_summary.csv"), index=False)
#                 pd.DataFrame(time_rows).to_csv(os.path.join(out_dir, "ctde_td_time_summary.csv"), index=False)
#
#     print("✅ All CTDE experiments with manual hyperparams are done.")

# if __name__ == "__main__":
#     # Set CPU affinity
#     try:
#         current_process = psutil.Process()
#         current_process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])
#         print(f"🧠 DQN Centralized training set to use CPU cores 0-7")
#     except Exception as e:
#         print(f"⚠️ Could not set CPU affinity: {e}")
#
#     os.nice(5)
#         # hyperparameter tuning
#     # gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#     # dqn_hyperparameter_tuning_per_gamma(gamma_values, num_eval_seeds=1, use_loss=True)
#
#     # # # Best hyperparameters for each gamma
#     best_hyperparams = {
#         0.1: {"learning_rate": 0.001, "hidden_dim": 64 , "hidden_layers": 2},
#         0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
#         0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
#         0.4: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 4},
#         0.5: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 4},
#         0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 3},
#     }
#
#     total_students_list = [100]
#     num_classrooms_list = [3]
#     action_levels_list = [3]
#     alphas = [0.0]
#     num_seeds = 10
#     max_steps = 73
#     gamma_values = list(best_hyperparams.keys())
#
#     for total_students in total_students_list:
#         for num_classrooms in num_classrooms_list:
#             for action_levels in action_levels_list:
#                 out_dir = (
#                     f"./results/centralized_td/"
#                     f"all_gamma_ts{total_students}_nc{num_classrooms}_al{action_levels}"
#                 )
#                 os.makedirs(out_dir, exist_ok=True)
#
#                 print(f"📦 Running centralized sweep over gamma in folder: {out_dir}")
#                 global_rows, ia_rows, time_rows = [], [], []
#
#                 for gamma in gamma_values:
#                     hyp = best_hyperparams[gamma]
#                     lr, hd, hl = hyp["learning_rate"], hyp["hidden_dim"], hyp["hidden_layers"]
#
#                     print(f"→ CENTRALIZED-TD | γ={gamma} | lr={lr} | hd={hd} | hl={hl} | students={total_students}")
#
#                     global_rows, ia_rows, time_rows = run_experiment_centralized(
#                         learning_rate=lr,
#                         hidden_dim=hd,
#                         hidden_layers=hl,
#                         total_students=total_students,
#                         num_classrooms=num_classrooms,
#                         action_levels=action_levels,
#                         gamma_values=[gamma],
#                         alphas=alphas,
#                         method="td",  # or "td"
#                         num_seeds=num_seeds,
#                         max_steps=max_steps,
#                         out_dir=out_dir,
#                         global_rows=global_rows,
#                         ia_rows=ia_rows,
#                         time_rows=time_rows
#                     )
#
#                 # Save once for all gamma values
#                 pd.DataFrame(global_rows).to_csv(os.path.join(out_dir, "centr_td_global_summary.csv"), index=False)
#                 pd.DataFrame(ia_rows).to_csv(os.path.join(out_dir, "centr_td_inf_allowed_summary.csv"), index=False)
#                 pd.DataFrame(time_rows).to_csv(os.path.join(out_dir, "centr_td_time_summary.csv"), index=False)
#
#     print("✅ All centralized experiments complete.")


# if __name__ == "__main__":
#     try:
#         current_process = psutil.Process()
#         current_process.cpu_affinity([0,1,2,3,4,5,6,7])
#         print(f"🧠 Myopic training set to use CPU cores 0-7")
#     except Exception as e:
#         print(f"⚠️ Could not set CPU affinity: {e}")
#
#     os.nice(5)
#
#     # Hyperparameters for myopic - gamma only matters here
#     gamma_values = [0.5]
#     alphas = [0.0]
#
#     total_students_list = [100]
#     num_classrooms_list = [2]
#     action_levels_list = [3]
#     num_seeds = 10
#     max_steps = 73
#
#     for total_students in total_students_list:
#         for num_classrooms in num_classrooms_list:
#             for action_levels in action_levels_list:
#                 out_dir = (
#                     f"./results/myopic/"
#                     f"all_gamma_ts{total_students}_nc{num_classrooms}_al{action_levels}_sf0.6"
#                 )
#                 os.makedirs(out_dir, exist_ok=True)
#
#                 print(f"📦 Running Myopic sweep over gamma in one folder: {out_dir}")
#
#                 global_rows, ia_rows = run_myopic_experiment(
#                     total_students=total_students,
#                     num_classrooms=num_classrooms,
#                     action_levels=action_levels,
#                     gamma_values=gamma_values,
#                     alphas=alphas,
#                     num_seeds=num_seeds,
#                     max_steps=max_steps,
#                     out_dir=out_dir,
#                 )
#
#     print("✅ All Myopic experiments are done.")
