import time
import datetime
import gym
import d4rl
import os
import random
import importlib
import numpy as np
import torch
import csv
import pickle 

from tqdm import tqdm
from util import plot_average_reward
from statistics import fmean
from sklearn import preprocessing

from models.tf_dynamics_models.constructor import construct_model
from models.tf_dynamics_models.utils import normalize
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from sac import SACPolicy
from mopo import MOPO
from buffer import ReplayBuffer
from logger import Logger
from fqe import FQE
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Trainer:
    def __init__(
        self,
        algo,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        traj_max_step,
        traj_num,
        eval_episodes=10
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self.traj_max_step = traj_max_step
        self.traj_num = traj_num
        self._eval_episodes = eval_episodes
        self.update_BO_traj_returns = []
        self.result_path = None

    def train_dynamics(self, load_model, result_path):
        self.result_path = result_path
        loss, total_model_num = self.algo.learn_dynamics(load_model, result_path)
        if not load_model:
            self.algo.save_dynamics_model(
                # save_path=os.path.join(self.logger.writer.get_logdir(), "dynamics_model")
                save_path = result_path+'/Trained_Dynamics_Model'
            )
        return loss, total_model_num
    
    def train_policy(self, exp_name, need_offline_eval=False):
        start_time = time.time()

        if self.algo._train_adversarial:
            self.algo.init_adversarial_model_update()

        num_timesteps = 0
        # train loop
        for e in range(1, self._epoch + 1):

            self.algo.policy.train()

            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    if num_timesteps % self._rollout_freq == 0:
                        self.algo.rollout_transitions()
                    
                    # update policy by sac
                    loss = self.algo.learn_policy()

                    t.set_postfix(**loss)

                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    
                    num_timesteps += 1
                    t.update(1)

            if self.algo._train_adversarial:
                self.algo.train_adversarial_model()

            # evaluate current policy
            if e == self._epoch:
                self._eval_episodes = 500
            else :
                self._eval_episodes = 1

            if need_offline_eval:
                eval_info = self._evaluate(offline_eval=False)
                _ = self._evaluate(offline_eval=True)
            else:
                eval_info = self._evaluate(offline_eval=False)

            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)
            self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}, episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")

            # save policy
            # torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "policy.pth"))

        self.logger.print('\033[33;1m' + '{} Average Reward: {:.3f}'.format(exp_name, ep_reward_mean) + '\033[33;0m')
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

        rewards = np.array(eval_info["eval/episode_reward"])
        plot_average_reward(x=list(range(self._eval_episodes)), y=rewards, x_label= 'episodes', y_label='rewards', y_min=-100, y_max=5000, 
                    title='Average Reward = {} ({})'.format(np.mean(rewards), exp_name), 
                    path=self.result_path+'/Trained_Policy/Rewards of {}'.format(exp_name))
    
        # save adversarial updated model
        if self.algo._train_adversarial:
            self.algo.save_dynamics_model(None)

        return fmean(self.update_BO_traj_returns), ep_reward_mean

    def save_policy(self, policy_name):
        torch.save(self.algo.policy.state_dict(), self.result_path+'/Trained_Policy/SAC_'+policy_name+".pth")

    def _evaluate(self, offline_eval=False):
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        step_count = 0
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if self.algo._train_adversarial:
            obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            if offline_eval:
                next_obs, reward, terminal, _ = self.algo.fake_env.step(obs, action, deterministic=True)
            else:
                next_obs, reward, terminal, _ = self.eval_env.step(action)
                if self.algo._train_adversarial:
                    next_obs = normalize(next_obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)
            episode_reward += reward
            episode_length += 1

            obs = next_obs
            step_count+=1

            if terminal or (step_count>self._step_per_epoch):
                if len(self.update_BO_traj_returns)==5:
                    self.update_BO_traj_returns.pop(0)
                self.update_BO_traj_returns.append(episode_reward)

                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                step_count = 0
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
                if self.algo._train_adversarial:
                    obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def get_predictions(self, observations, actions):
        next_obs, rewards, terminals, infos = self.algo.fake_env.step(observations, actions, deterministic=True)
        
        preds = np.concatenate((next_obs, rewards), axis=1)
        
        return preds.tolist()

    def get_trajectories(self, policy_ind, init_obs, random=False):
        trajectories=[]
        
        self.algo.policy.load_state_dict(torch.load(self.result_path+'/Trained_Policy/SAC_{}.pth'.format(policy_ind)))
        self.algo.policy.eval()
        
        for i in range(self.traj_num):
            obs = init_obs[i]
            step_count = 0
            terminal = False
            
            while step_count<self.traj_max_step:
                if terminal:
                    trajectories.append(np.concatenate((next_obs.flatten(), reward.flatten()), axis=0).tolist())
                    step_count+=1
                    
                else:
                    if random:
                        action = self.eval_env.action_space.sample()
                    else:
                        action = self.algo.policy.sample_action(obs, deterministic=True)
                        
                    next_obs, reward, terminal, _ = self.algo.fake_env.step(obs, action, deterministic=True)
                    trajectories.append(np.concatenate((next_obs.flatten(), reward.flatten()), axis=0).tolist())
                    
                    obs = next_obs
                    step_count+=1

        return trajectories


    def online_policy_evaluation(self, policy_ind, exp_name):
        self.algo.policy.load_state_dict(torch.load(self.result_path+'/Trained_Policy/SAC_{}.pth'.format(policy_ind)))
        self._eval_episodes = 500
            
        self.update_BO_traj_returns = []
        eval_info = self._evaluate()
        
            
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        
        self.logger.print('\033[33;1m' + 'Average Reward: {:.3f}'.format(ep_reward_mean) + '\033[33;0m')
        
        rewards = np.array(eval_info["eval/episode_reward"])
        plot_average_reward(x=list(range(self._eval_episodes)), y=rewards, x_label= 'episodes', y_label='rewards', y_min=-100, y_max=5000, 
                    title='Average Reward = {} ({})'.format(np.mean(rewards), exp_name), 
                    path=self.result_path+'/Trained_Policy/Rewards of {}'.format(exp_name))
        
        return fmean(self.update_BO_traj_returns), ep_reward_mean
    
    def offpolicy_evaluation(self, policy_ind, obs_dim, act_dim):
        eval_episodes = 500
        step_count = 0
        num_episodes = 0
        ep_eval_value = 0
        eval_values = []
        
        self.algo.policy.load_state_dict(torch.load(self.result_path+'/Trained_Policy/SAC_{}.pth'.format(policy_ind)))
        fqe = FQE(self.algo.policy, obs_dim, act_dim, self.result_path)
        if not os.path.isfile(self.result_path+'/OPE/fqe_{}.pth'.format(policy_ind)):
            fqe.train(self.algo.offline_buffer)
            torch.save(fqe.critic.state_dict(), self.result_path+'/OPE/fqe_{}.pth'.format(policy_ind))
        else:
            fqe.load(self.result_path+'/OPE/fqe_{}.pth'.format(policy_ind))
        
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        if self.algo._train_adversarial:
            obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        while num_episodes < eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, _, terminal, _ = self.algo.fake_env.step(obs, action, deterministic=True)
            
            value = fqe.value(obs)
            
            obs = next_obs
            step_count += 1
            ep_eval_value += value

            if terminal or (step_count>self._step_per_epoch):
                eval_values.append(ep_eval_value)
                num_episodes += 1
                step_count = 0
                ep_eval_value = 0
                obs = self.eval_env.reset()
                if self.algo._train_adversarial:
                    obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        ope = (sum(eval_values) / len(eval_values)).item()
        
        with open(self.result_path+'/Trained_Policy/OPE Result.csv', 'a', newline='') as csvfile:
            fieldnames = ['Policy', 'OPE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if os.stat(self.result_path+'/Trained_Policy/OPE Result.csv').st_size == 0 :
                writer.writeheader()
            writer.writerow({'Policy': policy_ind, 'OPE': ope})

        return ope
    
    
    def find_return(self, policy_ind, result_path):
        self.algo.policy.load_state_dict(torch.load(result_path+'/Trained_Policy/SAC_{}.pth'.format(policy_ind)))
        self.algo.policy.eval()
        
        self._eval_episodes = 500
            
        eval_ep_info_buffer = []
        step_count = 0
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        
        self.eval_env.seed(num_episodes)
        obs = self.eval_env.reset()
        print('->init',obs)
        if self.algo._train_adversarial:
            obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            if self.algo._train_adversarial:
                next_obs = normalize(next_obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)
            episode_reward += reward
            episode_length += 1

            obs = next_obs
            step_count+=1

            if terminal or (step_count>self._step_per_epoch):

                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                step_count = 0
                episode_reward, episode_length = 0, 0
                
                self.eval_env.seed(num_episodes)
                obs = self.eval_env.reset()
                if self.algo._train_adversarial:
                    obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        eval_info = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
            
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        rewards = eval_info["eval/episode_reward"]
        
    
        exp_info = []
        exp_info.append(policy_ind)
        exp_info.append(ep_reward_mean)
        for r in rewards:
            exp_info.append(round(r, 2))
        
        return exp_info
    
    
    def certain_BO_return(self, policy_ind, result_path):
        self.algo.policy.load_state_dict(torch.load(result_path+'/Trained_Policy/SAC_{}.pth'.format(policy_ind)))
        self.algo.policy.eval()
        
        eval_ep_info_buffer = []
        self.update_BO_traj_returns = []
        seed = [79, 463, 81, 27, 3, 212, 44, 345, 334, 412, 302, 139, 366, 154, 171, 383, 59, 495, 468, 63]
        
        self._eval_episodes = len(seed)
        step_count = 0
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        
        self.eval_env.seed(seed[num_episodes])
        obs = self.eval_env.reset()
        if self.algo._train_adversarial:
            obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            if self.algo._train_adversarial:
                next_obs = normalize(next_obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)
            episode_reward += reward
            episode_length += 1

            obs = next_obs
            step_count+=1

            if terminal or (step_count>self._step_per_epoch):
                self.update_BO_traj_returns.append(episode_reward)
                
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                step_count = 0
                episode_reward, episode_length = 0, 0
                
                if num_episodes < self._eval_episodes:
                    self.eval_env.seed(seed[num_episodes])
                    obs = self.eval_env.reset()
                    if self.algo._train_adversarial:
                        obs = normalize(obs, self.algo.offline_buffer.obs_mean, self.algo.offline_buffer.obs_std)

        eval_info = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
            
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        rewards = eval_info["eval/episode_reward"]
        
        return fmean(self.update_BO_traj_returns)


def create_trainer(args, load_model, model_name):
    env = gym.make(args.task)
    
    dataset_file = args.dataset_file
    if dataset_file is None:
        dataset = d4rl.qlearning_dataset(env)
    else:
        with open('datasets/'+ dataset_file+'.pkl', 'rb') as f:
            dataset = pickle.load(f)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"), 
        output_dim=args.action_dim,
        unbounded=True, 
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)    

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )

    if load_model:
        if args.dataset_file is None:
            model_dir = 'Experiment_Result/{}/Seed{}/Trained_Dynamics_Model'.format(args.task,args.seed)
        else:
            model_dir = 'Experiment_Result/pen-mixed/Seed{}/Trained_Dynamics_Model'.format(args.seed)
    else:
        model_dir = None

    # create dynamics model
    dynamics_model = construct_model(
        obs_dim=np.prod(args.obs_shape),
        act_dim=args.action_dim,
        hidden_dim=200,
        num_networks=args.n_ensembles,
        num_elites=args.n_elites,
        model_type="mlp",
        separate_mean_var=True,
        load_dir=model_dir,
        name=model_name
    )

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset, args.rambo, args.task)
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )

    # create MOPO algo
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    algo = MOPO(
        sac_policy,
        dynamics_model,
        static_fns=static_fns,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        rollout_batch_size=args.rollout_batch_size,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        train_adversarial=args.rambo,
    )

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer)

        
    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        traj_max_step=args.traj_max_step,
        traj_num=args.traj_num,
        eval_episodes=args.eval_episodes,
    )
    return trainer
