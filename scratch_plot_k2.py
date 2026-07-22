import os
import glob
import ppo_ctde
from ppo_ctde import MAPPO_CTDE, plot_policy_strips, OMEGA_VALUES

# Parameters for k=2
num_classrooms = 2
total_population = 100
shared_fraction = 0.3
OUTPUT_DIR = "mappo_results"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# Update global variables in ppo_ctde so plotting works correctly
ppo_ctde.NUM_CLASSROOMS = num_classrooms
ppo_ctde.TOTAL_POPULATION = total_population
ppo_ctde.TOTAL_STUDENTS = total_population // num_classrooms

representative_agents = {}
for omega in OMEGA_VALUES:
    # Find the most recently trained model for this omega and k=2
    search_pattern = f"mappo_omega_{omega}_sf_{shared_fraction}_k_{num_classrooms}_*_run_0.pt"
    matching_files = glob.glob(os.path.join(MODEL_DIR, search_pattern))
    
    if matching_files:
        # Get the latest file if multiple exist
        latest_model = max(matching_files, key=os.path.getctime)
        print(f"Loading {os.path.basename(latest_model)} for omega {omega}")
        
        # Load agent (strip .pt extension as required by the load method)
        model_path_no_ext = latest_model[:-3]
        mappo_agent = MAPPO_CTDE.load(model_path_no_ext)
        representative_agents[omega] = mappo_agent
    else:
        print(f"WARNING: Could not find model for omega {omega}")

if len(representative_agents) == len(OMEGA_VALUES):
    print("All models loaded successfully. Generating plot...")
    plot_policy_strips(representative_agents, num_classrooms)
    print(f"Plot saved! Check {OUTPUT_DIR}/combined_mappo_optimal_policies_k_{num_classrooms}.png")
else:
    print("Not all models were found. Cannot generate complete plot.")
