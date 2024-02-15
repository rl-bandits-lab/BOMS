import json
import math
import gym

from model_dist import ModelDist
from bayesian_opt.bayesian_optimization import BayesianOptimization
from bayesian_opt.util import UtilityFunction

class SelectionMethod():
    def __init__(self, args, trainers, init_model_ind, last_model_ind, init_obs, policy_df, result_path): 
        self.args = args
        self.trainers = trainers
        self.init_model_ind = init_model_ind
        self.last_model_ind = last_model_ind
        self.init_obs = init_obs
        self.policy_df = policy_df
        self.result_path = result_path

    
    def boms(self, n_exp):
        # Use BO to do the model selection
        best_bo_return = -math.inf
        curr_true_return = 0
        boms_opt_returns = []
        ep_returns = []
        bo_returns = []
        selected_indexes = []
        
        get_md = ModelDist(self.trainers, self.init_model_ind, self.init_obs)
        
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        model_selection_opt = BayesianOptimization(
            f=None,
            pbounds={'model_ind':(self.init_model_ind, self.last_model_ind)},
            result_path = self.result_path,
            init_model_ind = self.init_model_ind,
            last_model_ind = self.last_model_ind,
            verbose = 2,
            random_state = n_exp)

        
        for i in range(self.args.bo_update_times):
            print('[ BOMS ] Selection Epoch:{}'.format(i+1))
            selected_point = model_selection_opt.suggest(utility)
            selected_dynamic_ind = selected_point['model_ind']

            # trainer = create_trainer(self.args, load_model=True, model_name='BNN_{}'.format(selected_dynamic_ind))
            # trainer.train_dynamics(load_model=True, result_path=result_path)
            # partial_return_mean, ep_return_mean = trainer.train_policy(exp_name='{}-Exp {}-Updates of BOMS Online - Policy {}'.format(n_exp+1, i+1, selected_dynamic_ind))
            # trainer.save_policy(policy_name='{}'.format(selected_dynamic_ind))
            
            # partial_return_mean, ep_return_mean = trainer.online_policy_evaluation(selected_dynamic_ind, '{}-Exp {}-Updates of BOMS Online - Policy {}'.format(n_exp+1, i+1, selected_dynamic_ind), self.exp_path)
            
            cur_policy = self.policy_df.loc[self.policy_df['Policy'] == str(selected_dynamic_ind)]
            ep_return_mean = cur_policy.iloc[0]['True_Returns']
            partial_return_mean = cur_policy.iloc[0]['BO_Returns']
            
            model_selection_opt.register(params=selected_point, target=partial_return_mean)

            ep_returns.append(round(ep_return_mean, 2))
            bo_returns.append(round(partial_return_mean, 2))
            selected_indexes.append(selected_dynamic_ind)
            
            model_dists = get_md.get_model_dist(selected_dynamic_ind)
            
            with open(self.result_path+"/model_distances", "w") as fp:
                json.dump(model_dists, fp)

            if partial_return_mean > best_bo_return:
                best_bo_return = partial_return_mean
                curr_true_return = ep_return_mean
            boms_opt_returns.append(round(curr_true_return, 2))
            
        
        with open(self.result_path+'/Results/BOMS {}'.format(n_exp+1)+'.txt', 'w') as f:
            f.write('Selected Model Indexes: '+str(selected_indexes)+'\n')
            f.write('Trajectory Optimal Returns for Episode: '+str(boms_opt_returns)+'\n')

        return model_selection_opt.max


    def ablation(self, ablation, n_exp):
        best_bo_return = -math.inf
        curr_true_return = 0
        boms_opt_returns = []
        ep_returns = []
        bo_returns = []
        selected_indexes = []
        
        get_md = ModelDist(self.trainers, self.init_model_ind, self.init_obs)
        
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        model_selection_opt = BayesianOptimization(
            f=None,
            pbounds={'model_ind':(self.init_model_ind, self.last_model_ind)},
            result_path = self.result_path,
            init_model_ind = self.init_model_ind,
            last_model_ind = self.last_model_ind,
            verbose = 2,
            random_state = n_exp)
        
        
        for i in range(self.args.bo_update_times):
            print('[ Ablation | {} ] Selection Epoch: {}'.format(ablation, i+1)) 
            selected_point = model_selection_opt.suggest(utility)
            selected_dynamic_ind = selected_point['model_ind']

            cur_policy = self.policy_df.loc[self.policy_df['Policy'] == str(selected_dynamic_ind)]
            ep_return_mean = cur_policy.iloc[0]['True_Returns']
            partial_return_mean = cur_policy.iloc[0]['BO_Returns']

            model_selection_opt.register(params=selected_point, target=partial_return_mean)

            ep_returns.append(round(ep_return_mean, 2))
            bo_returns.append(round(partial_return_mean, 2))
            selected_indexes.append(selected_dynamic_ind)

            if ablation == 'Explo Policy':
                model_dists = get_md.explo_dist(selected_dynamic_ind)
            if ablation == 'Trained Policy':
                model_dists = get_md.trained_dist(selected_dynamic_ind)
            if ablation == 'Weight Bias':
                model_dists = get_md.weight_dist(selected_dynamic_ind, self.result_path)
            if ablation == 'Behavior Policy':
                model_dists = get_md.behavior_dist(selected_dynamic_ind)
                
            
            with open(self.result_path+'/model_distances', "w") as fp:
                json.dump(model_dists, fp)

            if partial_return_mean > best_bo_return:
                best_bo_return = partial_return_mean
                curr_true_return = ep_return_mean
            boms_opt_returns.append(round(curr_true_return, 2))
            
        
        with open(self.result_path+'/Results/Ablation - {} {}'.format(ablation, n_exp+1)+'.txt', 'w') as f:
            f.write('Selected Model Indexes: '+str(selected_indexes)+'\n')
            f.write('Trajectory Optimal Returns for Episode: '+str(boms_opt_returns)+'\n')

        return model_selection_opt.max


    def random(self, n_exp):
        best_partial_return = -math.inf
        curr_true_return = 0
        urms_best_returns = []
        
        model_selection_opt = BayesianOptimization(
            f=None,
            pbounds={'model_ind':(self.init_model_ind, self.last_model_ind)},
            result_path = self.result_path,
            init_model_ind = self.init_model_ind,
            last_model_ind = self.last_model_ind,
            verbose = 2,
            random_state = n_exp)
        
        rand_indexes = model_selection_opt.random_points(self.args.bo_update_times)
        
        for i, index in enumerate(rand_indexes):
            print('[ Baseline | Random ] Selection Epoch: {}'.format(i+1))
            selected_dynamic_ind = index

            cur_policy = self.policy_df.loc[self.policy_df['Policy'] == str(selected_dynamic_ind)]
            ep_return_mean = cur_policy.iloc[0]['True_Returns']
            partial_return_mean = cur_policy.iloc[0]['BO_Returns']
            
            if partial_return_mean > best_partial_return:
                best_partial_return = partial_return_mean
                curr_true_return = ep_return_mean
            urms_best_returns.append(round(curr_true_return, 2))
        
        with open(self.result_path+'/Results/Baseline - Random Selection {}'.format(n_exp+1)+'.txt', 'w') as f:
            f.write('Index: '+str(rand_indexes)+'\n')
            f.write('Trajectory Optimal Returns for Episode: '+str(urms_best_returns)+'\n')


    def mopo(self):
        mopo_returns = []

        cur_policy = self.policy_df.loc[self.policy_df['Policy'] == 'MOPO']
        partial_return_mean = cur_policy.iloc[0]['BO_Returns']
        ep_return_mean = cur_policy.iloc[0]['True_Returns']
        
        for i in range(self.args.bo_update_times):
            mopo_returns.append(round(ep_return_mean, 2))
        
        with open(self.result_path+'/Results/Baseline - MOPO.txt', 'w') as f:
            f.write('Trajectory Optimal Returns for Episode: '+str(mopo_returns)+'\n')


    def boms_offline(self, n_exp):
        best_bo_return = -math.inf
        curr_true_return = 0
        boms_true_returns = []
        selected_indexes = []

        get_md = ModelDist(self.trainers, self.init_model_ind, self.init_obs)
        
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        model_selection_opt = BayesianOptimization(
            f=None,
            pbounds={'model_ind':(self.init_model_ind, self.last_model_ind)},
            result_path = self.result_path,
            init_model_ind = self.init_model_ind,
            last_model_ind = self.last_model_ind,
            verbose=2,
            random_state=n_exp)

        for i in range(self.args.bo_update_times):
            print('[ Baseline | BOMS Offline ] Selection Epoch: {}'.format(i+1))
            next_point = model_selection_opt.suggest(utility)
            selected_dynamic_ind = next_point['model_ind']

            cur_policy = self.policy_df.loc[self.policy_df['Policy'] == str(selected_dynamic_ind)]
            partial_return_mean = cur_policy.iloc[0]['OPE']
            ep_return_mean = cur_policy.iloc[0]['True_Returns']
            
            model_selection_opt.register(params=next_point, target=partial_return_mean)

            model_dists = get_md.get_model_dist(selected_dynamic_ind)
            with open(self.result_path+"/model_distances", "w") as fp:
                json.dump(model_dists, fp)

            if partial_return_mean > best_bo_return:
                best_bo_return = partial_return_mean
                curr_true_return = ep_return_mean
            boms_true_returns.append(round(curr_true_return, 2))
            selected_indexes.append(selected_dynamic_ind)

        with open(self.result_path+'/Results/Baseline - BOMS(offline) {}'.format(n_exp+1)+'.txt', 'w') as f:
            f.write('Selected Model Indexes: '+str(selected_indexes)+'\n')
            f.write('Trajectory Optimal Returns for Episode: '+str(boms_true_returns)+'\n')