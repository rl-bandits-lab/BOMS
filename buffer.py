import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        obs_shape,
        obs_dtype,
        action_dim,
        action_dtype,
    ):
        self.max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, terminal):
        # Copy to avoid modification by reference
        self.observations[self.ptr] = np.array(obs).copy()
        self.next_observations[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.terminals[self.ptr] = np.array(terminal).copy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def load_dataset(self, dataset, normalization, dataset_name=""):
        self.obs_mean, self.obs_std = None, None
        if normalization:
            dataset, self.obs_mean, self.obs_std = self._normalise_data(dataset, True, False, dataset_name)

        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"]).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self.ptr = len(observations)
        self.size = len(observations)

    def _normalise_data(self, data, normalize_states, normalize_rewards, dataset_name):
        obs_mean = None
        obs_std = None
        if (not normalize_states) and (not normalize_rewards):
            return data, obs_mean, obs_std

        obs = data["observations"]
        next_obs = data["next_observations"]
        rewards = data["rewards"]

        # compute mean and std across subsample of data
        inds = np.floor(np.linspace(0, obs.shape[0]-1, num=10000)).astype(int)

        # subtract 1 from antmaze rewards per IQL paper
        if 'antmaze' in dataset_name:
            data['rewards'] -= 1

        if normalize_rewards:
            rew_std = np.std(rewards[inds, :], axis=0) + 1e-6
            data['rewards'] = rewards / rew_std

        if normalize_states:
            obs_mean = np.mean(obs[inds, :], axis=0)
            obs_std = np.std(obs[inds, :], axis=0) + 1e-6 # avoid division by zero
            obs_norm = (obs - obs_mean) / obs_std
            next_obs_norm = (next_obs - obs_mean) / obs_std

            data["observations"] = obs_norm
            data["next_observations"] = next_obs_norm

        return data, obs_mean, obs_std
    
    def add_batch(self, obs, next_obs, actions, rewards, terminals):
        batch_size = len(obs)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.observations[begin:end] = np.array(obs[:first_add_size]).copy()
            self.next_observations[begin:end] = np.array(next_obs[:first_add_size]).copy()
            self.actions[begin:end] = np.array(actions[:first_add_size]).copy()
            self.rewards[begin:end] = np.array(rewards[:first_add_size]).copy()
            self.terminals[begin:end] = np.array(terminals[:first_add_size]).copy()

            begin = 0
            end = batch_size - first_add_size
            self.observations[begin:end] = np.array(obs[first_add_size:]).copy()
            self.next_observations[begin:end] = np.array(next_obs[first_add_size:]).copy()
            self.actions[begin:end] = np.array(actions[first_add_size:]).copy()
            self.rewards[begin:end] = np.array(rewards[first_add_size:]).copy()
            self.terminals[begin:end] = np.array(terminals[first_add_size:]).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.observations[begin:end] = np.array(obs).copy()
            self.next_observations[begin:end] = np.array(next_obs).copy()
            self.actions[begin:end] = np.array(actions).copy()
            self.rewards[begin:end] = np.array(rewards).copy()
            self.terminals[begin:end] = np.array(terminals).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):

        batch_indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy()
        }
    
    def sample_all(self):
        return {
            "observations": self.observations[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "next_observations": self.next_observations[:self.size].copy(),
            "terminals": self.terminals[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy()
        }
