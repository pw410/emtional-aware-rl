# Emotion-Aware Reinforcement Learning (EA-RL)

This repository contains the official implementation of **Emotion-Aware Reinforcement Learning (EA-RL)** —  
a hybrid conscious–subconscious control model integrating emotional reward shaping,  
awareness-based priority gating, and Ramanujan nonlinear transforms.

EA-RL integrates:
- Emotional Reward Module  
- Awareness & Priority Gating System  
- Subconscious Emotional Memory Weights  
- Ramanujan Radical Nonlinear Transforms  
- 10 fully reproducible experiments  
- Plotting + table generation scripts  

---

##  Repository Structure

emtional-aware-rl/  
│  
├── **src/**  
│   ├── grid_env.py  
│   ├── dqn_agent.py  
│   ├── awareness_module.py  
│   ├── emotion_module.py  
│   ├── emotion_module_exp2.py  
│   ├── emotion_module_exp3.py  
│   ├── emotion_module_exp4.py  
│   ├── emotion_module_exp5.py  
│   ├── emotion_module_exp6.py  
│   ├── emotion_module_exp7.py  
│   ├── train_ex1.py  
│   ├── train_ex2.py  
│   ├── train_ex3.py  
│   ├── train_ex4.py  
│   ├── train_ex5.py  
│   ├── train_ex6.py  
│   ├── train_ex7.py  
│   ├── train_ex8.py  
│   ├── train_ex9.py  
│   ├── train_ex10.py  
│   ├── plot_ex1.py  
│   ├── plot_ex2.py  
│   ├── plot_ex3.py  
│   ├── plot_ex4.py  
│   ├── plot_ex5.py  
│   ├── plot_ex6.py  
│   ├── plot_ex7.py  
│   ├── plot_ex8.py  
│   ├── plot_ex9.py  
│   ├── plot_ex10.py  
│   ├── ex1_table.py  
│   └── .keep  
│  
├── LICENSE  
└── README.md  

---

##  How to Run Experiments

Navigate to the source directory:

cd emtional-aware-rl/src

Run any experiment:

python train_ex1.py python train_ex2.py python train_ex3.py ... python train_ex10.py

Each script automatically:
- Creates the GridWorld environment  
- Loads emotional + awareness modules  
- Trains the EA-RL agent  
- Saves results as `.pkl` files  

---

##  Generate Plots / Figures

For any experiment:

python plot_ex1.py python plot_ex2.py ... python plot_ex10.py

Plots are automatically saved to:

graphs_all/

---

##  Generate Supplementary Tables

Example:

python ex1_table.py

This generates a clean table summarizing Experiment-1 metrics.

---

##  Requirements

Python ≥ 3.9
numpy
matplotlib
pickle
gym

Works on any standard Python environment.

---

##  Citation

(To be added after paper acceptance.)


---
