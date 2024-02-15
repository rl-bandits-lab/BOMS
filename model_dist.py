import numpy as np
from scipy.spatial.distance import euclidean
from scipy.io import loadmat
import os

class ModelDist():
    def __init__(self, trainers, init_model_ind, init_obs=None): 
        self.trainers = trainers
        self.num = len(trainers)
        self.init_ind = init_model_ind
        self.init_obs = init_obs
        
        self.trained_dists_matrix = np.zeros((self.num,self.num))
        self.trained_visited = np.zeros((self.num,self.num))
        self.behavior_dists_matrix = np.zeros((self.num,self.num))
        self.behavior_visited = np.zeros((self.num,self.num))
        self.explo_dists_matrix = np.zeros((self.num,self.num))
        self.explo_visited = np.zeros((self.num,self.num))
        self.sample_dists_matrix = np.zeros((self.num,self.num))
        self.sample_visited = np.zeros((self.num,self.num))
        self.samples = []
        self.weight_dists_matrix = np.zeros((self.num,self.num))
        self.traj_num = 10
        self.n_sample = 1000
        
        self.true_sample = []
        self.compare_dists_matrix = np.zeros((self.num,self.num))

    def get_model_dist(self, curr_ind):
        dist = 0
        _curr_ind = curr_ind-self.init_ind
        
        if len(self.samples)==0:
            transitions = self.trainers[_curr_ind].algo.offline_buffer.sample(self.n_sample)
            observations = transitions["observations"]
            actions = self.trainers[_curr_ind].algo.policy.sample_action(observations, deterministic=True)

            for i in range(self.num):
                preds = self.trainers[i].get_predictions(observations, actions)
                self.samples.append(preds)

        for _iter_ind in range(len(self.samples)):
            curr_sample = self.samples[_curr_ind]
            iter_sample = self.samples[_iter_ind]
            
            sample_len = len(curr_sample)
            dist = 0
            
            for t in range(sample_len):
                dist += euclidean(curr_sample[t], iter_sample[t])/ sample_len

            if self.sample_visited[_curr_ind][_iter_ind] == 0:
                self.sample_dists_matrix[_curr_ind][_iter_ind] = dist
                self.sample_visited[_curr_ind][_iter_ind] = 1

            if self.sample_visited[_iter_ind][_curr_ind] == 0:
                self.sample_dists_matrix[_iter_ind][_curr_ind] = dist
                self.sample_visited[_iter_ind][_curr_ind] = 1
                
        return self.sample_dists_matrix.tolist()
    
    
    def trained_dist(self, curr_ind):
        trajectories = []
        _curr_ind = curr_ind-self.init_ind
        
        for i in range(self.num):
            traj = self.trainers[i].get_trajectories(curr_ind, self.init_obs)
            trajectories.append(traj)

        for _iter_ind in range(len(trajectories)):
            curr_traj = trajectories[_curr_ind]
            iter_traj = trajectories[_iter_ind]
            
            traj_len = len(curr_traj)
            dist = 0
            
            for t in range(traj_len):
                dist += (euclidean(curr_traj[t], iter_traj[t]))/ traj_len
    
            if self.trained_visited[_curr_ind][_iter_ind] == 0:
                self.trained_dists_matrix[_curr_ind][_iter_ind] = dist
                self.trained_visited[_curr_ind][_iter_ind] = 1

            if self.trained_visited[_iter_ind][_curr_ind] == 0:
                self.trained_dists_matrix[_iter_ind][_curr_ind] = dist
                self.trained_visited[_iter_ind][_curr_ind] = 1
        
        return self.trained_dists_matrix.tolist()


    def behavior_dist(self, curr_ind):
        trajectories = []
        _curr_ind = curr_ind-self.init_ind
        
        for i in range(self.num):
            traj = self.trainers[i].get_trajectories('Behavior', self.init_obs)
            trajectories.append(traj)

        for _iter_ind in range(len(trajectories)):
            curr_traj = trajectories[_curr_ind]
            iter_traj = trajectories[_iter_ind]
            
            traj_len = len(curr_traj)
            dist = 0
            
            for t in range(traj_len):
                dist += (euclidean(curr_traj[t], iter_traj[t]))/traj_len
            
            if self.behavior_visited[_curr_ind][_iter_ind] == 0:
                self.behavior_dists_matrix[_curr_ind][_iter_ind] = dist
                self.behavior_visited[_curr_ind][_iter_ind] = 1

            if self.behavior_visited[_iter_ind][_curr_ind] == 0:
                self.behavior_dists_matrix[_iter_ind][_curr_ind] = dist
                self.behavior_visited[_iter_ind][_curr_ind] = 1
                
        return self.behavior_dists_matrix.tolist()

    
    def explo_dist(self, curr_ind):
        trajectories = []
        _curr_ind = curr_ind-self.init_ind
        
        for i in range(self.num):
            traj = self.trainers[i].get_trajectories(curr_ind, self.init_obs, random=True)
            trajectories.append(traj)

        for _iter_ind in range(len(trajectories)):
            curr_traj = trajectories[_curr_ind]
            iter_traj = trajectories[_iter_ind]
            
            traj_len = len(curr_traj)
            dist = 0
            
            for t in range(traj_len):
                dist += (euclidean(curr_traj[t], iter_traj[t]))/ traj_len
            
            if self.explo_visited[_curr_ind][_iter_ind] == 0:
                self.explo_dists_matrix[_curr_ind][_iter_ind] = dist
                self.explo_visited[_curr_ind][_iter_ind] = 1

            if self.explo_visited[_iter_ind][_curr_ind] == 0:
                self.explo_dists_matrix[_iter_ind][_curr_ind] = dist
                self.explo_visited[_iter_ind][_curr_ind] = 1
                
        return self.explo_dists_matrix.tolist()
        

    def weight_dist(self, curr_ind, result_path):
        curr_params_dict = loadmat(os.path.join(result_path+'/Trained_Dynamics_Model/BNN_{}.mat'.format(curr_ind)))
        _curr_ind = curr_ind-self.init_ind
        keys = list(curr_params_dict.keys())
        keys.remove('__header__')
        keys.remove('__version__')
        keys.remove('__globals__')

        for _iter_ind in range(self.num):
            iter_ind = _iter_ind+self.init_ind
            iter_params_dict = loadmat(os.path.join(result_path+'/Trained_Dynamics_Model/BNN_{}.mat'.format(iter_ind)))
            dist = 0

            for key in keys:
                curr_params = np.array(curr_params_dict[key]).flatten()
                iter_params = np.array(iter_params_dict[key]).flatten()
                
                dist += euclidean(curr_params, iter_params)
                
            self.weight_dists_matrix[_curr_ind][_iter_ind] = dist
            self.weight_dists_matrix[_iter_ind][_curr_ind] = dist

        return self.weight_dists_matrix.tolist()