# SafeCampus MultiClassroom Project

## Overview
This project implements a reinforcement learning approach to optimize classroom policies in a multi-classroom environment during a pandemic scenario. It uses two different RL algorithms: Q-Learning and Deep Q-Network (DQN).

## Project Structure
The project consists of the following main components:

- `environment/`: Contains the `MultiClassroomEnv` class, which simulates the multi-classroom environment.
- `agents/`: 
  - `q_learning.py`: Implements the `IndependentQLearningAgent` for tabular Q-learning.
  - `dqn_agent.py`: Implements the `DQNAgent` for deep Q-learning.
- `utils/`: Contains utility functions, including visualization tools.
- `main.py`: The main script to run experiments with both Q-learning and DQN.

## How to Use

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. By default, this will run the Q-learning algorithm. To run the DQN algorithm, uncomment the `dqn_run()` function call in the `main()` function.

4. Results and visualizations will be saved in the `results/` directory.

## Customization

You can customize various parameters in the `main.py` file:

- Number of classrooms
- Total number of students
- Number of shared students
- Maximum number of weeks
- Action levels per classroom
- Random seed

## Output

The script will output:
- Training progress
- Test results for specific state combinations
- Visualizations of Q-tables (for Q-learning) or learned policies (for DQN)

## Notes

- The current implementation uses discrete action and state spaces for q learning and continous states for DQN.
- The environment simulates the spread of infection within and between classrooms.
- The agents learn policies to minimize the spread of infection while balancing classroom attendance.



