# BOMS

This is the implementation for **Active Model Selection for Offline Model-Based RL via Bayesian Optimization**. We reuse [pytorch-mopo](https://github.com/yihaosun1124/pytorch-mopo) as the basis of our implementation.


Requirements
---
- D4RL
- Gym 0.22.0
- MuJoCo 2.3.5
- Python 3.8.10
- PyTorch 1.8+
- TensorFlow 2.x


Main Code Structure
---
```
BOMS/
├── bayesian_opt/ --- Bayesian Optimization package with model-induced kernels
├── models/ --- the structure of dynamics models and policies
├── model_dist.py --- various methods of model distance measuring
├── mopo.py --- the structure of MOPO algorithm
├── sac.py --- the structure of SAC
├── selection_method.py --- different model selection schemes
├── train.py --- the main file for BOMS algorithm execution
└── trainer.py ---  the execution file of dynamics models and policies 
```


Code Execution
---
To execute BOMS algorithm, run the `train.py` with: 
```
python3 train.py --task walker2d-medium-v0
```
, and the evaluation results will be displayed in the folder `Experiment_Result`.

Here are some main hyperparameters in BOMS that you can change:
- task: Tasks in D4RL benchmark dataset
- traj_num: The number of online trajectories for evaluation
- traj_max_step: The max steps of each online trajectory
- rambo: Integrating BOMS with RAMBO or not
  
