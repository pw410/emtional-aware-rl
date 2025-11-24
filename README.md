Emotion-Aware Reinforcement Learning (EA-RL)
A hybrid conscious–subconscious control model integrating emotional reward, awareness gating, and Ramanujan nonlinear transforms.

--->Overview
This repository contains the official implementation of Emotion-Aware Reinforcement Learning (EA-RL) —
a reinforcement-learning architecture designed to produce human-aligned, emotionally informed, safe, and stable decision-making.

EA-RL integrates:
Emotional Reward Module
Awareness & Priority Gating System
Subconscious Emotional Memory Weights
Ramanujan Radical Nonlinear Transforms
10 fully reproducible experiments + plotting scripts

-->Repository Structure

emotional-aware-rl/
│
├── src/                     # Core architecture
│   ├── grid_env.py
│   ├── dqn_agent.py
│   ├── awareness_module.py
│   ├── emotion_module.py
│   ├── emotion_module_exp2.py
│   ├── emotion_module_exp3.py
│   ├── emotion_module_exp4.py
│   ├── emotion_module_exp5.py
│   ├── train_ex1.py ... train_ex10.py
│   ├── plot_ex1.py  ... plot_ex10.py
│   ├── ex1_table.py
│   └── .keep
│
├── graphs_all/              # All supplementary graphs (PNG)
│
├── LICENSE
└── README.md


--> How to Run Experiments

Navigate to the code directory:
cd emotional-aware-rl/src
Run any experiment:

python train_ex1.py
python train_ex2.py
...
python train_ex10.py
Each script automatically:

Creates the GridWorld environment
Loads emotional + awareness modules
Trains the EA-RL agent
Saves results as .pkl files

--> Generate Plots / Figures

For any experiment:

python plot_ex1.py
python plot_ex2.py
...
...
python plot_ex10.py

Plots are automatically saved to:

graphs_all/


--> Generate Supplementary Tables

Example:
python ex1_table.py
This generates a clean table summarizing Experiment 1’s metrics.


-->Requirements
python >= 3.9
numpy
matplotlib
pickle
gym

Works on any standard Python environment.

-----------------------------------
 
✔ Exactly GitHub standard formatting

Paste this in README.md and click Commit changes.
