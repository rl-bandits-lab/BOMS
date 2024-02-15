import argparse
import json

import gym
import d4rl

import numpy as np
import torch
import csv
import os
import pandas as pd

from trainer import create_trainer
from util import merge_csv, plot_baseline_regrets, plot_ablation_regrets
from selection_method import SelectionMethod


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="boms")
    parser.add_argument("--task", type=str, default="walker2d-random-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--bo_update_times", type=int, default=10)
    parser.add_argument("--cands_model_num", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--exp_num", type=int, default=10)
    parser.add_argument("--traj_num", type=int, default=10)
    parser.add_argument("--traj_max_step", type=int, default=100)

    parser.add_argument("--rambo", action="store_true")

    return parser.parse_args()


# +
if __name__ == "__main__":
    args = get_args()
    env = gym.make(args.task)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    result_path = 'Experiment_Result/{}/Seed{}'.format(args.task, args.seed)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(result_path+'/OPE')
        os.makedirs(result_path+'/Results')
        os.makedirs(result_path+'/Trained_Policy')
        os.makedirs(result_path+'/Trained_Dynamics_Model')

    init_obs = []
    for _ in range(args.traj_num):
        obs = env.reset()
        init_obs.append(obs)


    """ Model Training """
    trainer = create_trainer(args, load_model=False, model_name=None)
    _, total_model_num = trainer.train_dynamics(load_model=False, result_path=result_path)

    args.cands_model_num = args.cands_model_num if total_model_num>args.cands_model_num else total_model_num
    last_model_ind = total_model_num-1
    init_model_ind = total_model_num-args.cands_model_num

    with open(result_path+'/Model Index Range.txt', 'w') as f:
        f.write('Index Range: ' + str(init_model_ind) + '-' + str(last_model_ind)+'\n')
        

    """ Policy Training """
    indexes = [*range(init_model_ind, last_model_ind+1), 'MOPO']

    for dynamic_ind in indexes:
        trainer = create_trainer(args, load_model=True, model_name='BNN_{}'.format(dynamic_ind))
        trainer.train_dynamics(load_model=True, result_path=result_path)
        partial_return_mean, ep_return_mean = trainer.train_policy(exp_name='{}-Seed - Policy {}'.format(args.seed, dynamic_ind))
        trainer.save_policy(policy_name='{}'.format(dynamic_ind))
        
        trainer.offpolicy_evaluation(dynamic_ind, args.obs_shape[0], args.action_dim)
        
        with open(result_path+'/Trained_Policy/Policy Training Result.csv', 'a', newline='') as csvfile:
            fieldnames = ['Policy', 'BO_Returns', 'True_Returns']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if os.stat(result_path+'/Trained_Policy/Policy Training Result.csv').st_size == 0 :
                writer.writeheader()
            writer.writerow({'Policy':dynamic_ind, 'BO_Returns': partial_return_mean, 'True_Returns': ep_return_mean})

    args.real_ratio = 1
    trainer = create_trainer(args, load_model=True, model_name='BNN_MOPO')
    trainer.train_dynamics(load_model=True, result_path=result_path)
    partial_return_mean, ep_return_mean = trainer.train_policy(exp_name='{}-Seed - Behavior Policy'.format(args.seed))
    trainer.save_policy(policy_name='Behavior')
    args.real_ratio = 0.05

    merge_csv('Policy Training Result.csv', 'OPE Result.csv', result_path)

    
    """ Experiment """
    trainers = []
    init_model_ind = init_model_ind
    last_model_ind = last_model_ind
    policy_df = pd.read_csv(result_path+'/Trained_Policy/Policy Performance.csv')
    
    for i in range(args.cands_model_num):
        trainers.append(create_trainer(args, load_model=True, model_name='BNN_{}'.format(init_model_ind+i)))
        loss, _ = trainers[i].train_dynamics(load_model=True, result_path=result_path)
    
    selection = SelectionMethod(args, trainers, init_model_ind, last_model_ind, init_obs, policy_df, result_path)
    selection.mopo()
    for n_exp in range(args.exp_num):  
        selection.boms(n_exp)
        selection.random(n_exp)

        selection.ablation('Trained Policy', n_exp)
        selection.ablation('Explo Policy', n_exp)
        selection.ablation('Behavior Policy', n_exp)
        selection.ablation('Weight Bias', n_exp)

    plot_baseline_regrets(args, policy_df, args.bo_update_times, result_path)
    plot_ablation_regrets(args, policy_df, args.bo_update_times, result_path)  

    