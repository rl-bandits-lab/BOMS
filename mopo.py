import os

import numpy as np
import torch

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from models.tf_dynamics_models.fake_env import FakeEnv
from models.tf_dynamics_models.constructor import format_samples_for_training, construct_model
import models.tf_dynamics_models.utils as utils


class MOPO():
    def __init__(
        self,
        policy,
        dynamics_model,
        static_fns,
        offline_buffer,
        model_buffer,
        reward_penalty_coef,
        rollout_length,
        rollout_batch_size,
        batch_size,
        real_ratio,
        train_adversarial, # becomes RAMBO, if True
        include_entropy_in_adv = True,
        adversary_loss_weighting = 3e-4,
        adv_lr = 3e-4,
    ):

        self.policy = policy
        self.dynamics_model = dynamics_model
        self.static_fns = static_fns
        self.fake_env = FakeEnv(
            self.dynamics_model,
            self.static_fns,
            penalty_coeff=reward_penalty_coef,
            penalty_learned_var=True,
            obs_mean=offline_buffer.obs_mean,
            obs_std=offline_buffer.obs_std
        )
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._batch_size = batch_size
        self._real_ratio = real_ratio

        self._train_adversarial = train_adversarial
        self._include_entropy_in_adv = include_entropy_in_adv
        self._adversary_loss_weighting = adversary_loss_weighting
        self._adv_lr = adv_lr

    def _sample_initial_transitions(self):
        return self.offline_buffer.sample(self._rollout_batch_size)

    def rollout_transitions(self):
        init_transitions = self._sample_initial_transitions()

        # rollout
        observations = init_transitions["observations"]
        for _ in range(self._rollout_length):
            
            actions = self.policy.sample_action(observations)

            next_observations, rewards, terminals, infos = self.fake_env.step(observations, actions, deterministic=True)

            self.model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
        
            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

    def learn_dynamics(self, load_model, result_path):
        data = self.offline_buffer.sample_all()
        train_inputs, train_outputs = format_samples_for_training(data)
        max_epochs = 1 if self.dynamics_model.model_loaded else None
        loss, total_model_num = self.dynamics_model.train(
            train_inputs,
            train_outputs,
            batch_size=self._batch_size,
            max_epochs=max_epochs,
            holdout_ratio=0.2,
            load_model=load_model,
            result_path = result_path
        )
        return loss, total_model_num

    def learn_policy(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        loss = self.policy.learn(data)
        return loss
    
    def save_dynamics_model(self, save_path):
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        self.dynamics_model.save(save_path, type_name='MOPO')

    def train_adversarial_model(self):
        """ train adversarial model using on-policy updates.
        """

        alpha = self.policy._alpha.item() if self.policy._is_auto_alpha else self.policy._alpha

        steps = 0
        self._epoch_length = 1000
        while steps < self._epoch_length:
            batch = self.offline_buffer.sample(self._batch_size)
            obs = batch['observations']
            for t in range(self._rollout_length):
                act = self.policy.sample_action(obs)
                inputs, targets = self.dynamics_model.get_labeled_batch()

                sa = np.concatenate([obs, act], -1)
                ensemble_model_means, ensemble_model_vars = self.dynamics_model.predict(sa, factored=True)
                batch_size = self._batch_size

                # because model predicts deltas for observations add original obs
                rewards_means = ensemble_model_means[:, :, 0:1]
                ensemble_model_means = np.concatenate([rewards_means, ensemble_model_means[:,:,1:]+obs], -1)
                ensemble_model_stds = np.sqrt(ensemble_model_vars)
                random_normal = np.random.normal(size=(ensemble_model_means.shape[0], batch_size, ensemble_model_means.shape[2]))
                ensemble_samples = ensemble_model_means + random_normal * ensemble_model_stds

                # use one model from ensemble
                model_inds = self.dynamics_model.random_inds(batch_size).astype(int)
                model_inds = (model_inds, [i for i in range(len(model_inds))])
                samples = ensemble_samples[model_inds]
                rewards = np.squeeze(samples[:, :1])
                next_obs = samples[:, 1:]

                with torch.no_grad():
                    pred_Qs_value = torch.minimum(
                        self.policy.critic1(obs, act),
                        self.policy.critic2(obs, act)
                    ).cpu().numpy()

                    next_actions, policy_log_prob = self.policy(next_obs, True)
                    next_actions = next_actions.cpu().numpy()
                    policy_log_prob = policy_log_prob.cpu().numpy().squeeze()
                    next_Qs_values = torch.minimum(
                        self.policy.critic1(next_obs, next_actions),
                        self.policy.critic2(next_obs, next_actions)
                    ).cpu().numpy()

                min_next_Q = np.squeeze(next_Qs_values)

                # whether to include the entropy bonus at the next state in advantage calc
                if self._include_entropy_in_adv:
                    next_value = min_next_Q - alpha * policy_log_prob
                else:
                    next_value = min_next_Q

                # use terminals like mopo
                terminals = self.fake_env.config.termination_fn(
                    utils.unnormalize(obs, self.offline_buffer.obs_mean, self.offline_buffer.obs_std),
                    act,
                    utils.unnormalize(next_obs, self.offline_buffer.obs_mean, self.offline_buffer.obs_std)
                ).squeeze()
                done_mask = np.ones_like(terminals) - terminals.astype(int)
                value = rewards + self.policy._gamma * next_value * done_mask

                pred_value = np.squeeze(pred_Qs_value)

                # normalise advantages using batch mean and std
                advantages = value - pred_value
                advantages = (advantages - np.mean(advantages)) / np.std(advantages)

                feed_dict = {
                    self._observations_ph: obs,
                    self._actions_ph: act,
                    self._advantage_ph: advantages,
                    self._random_normal_ph: random_normal,
                    self._model_inds_ph: list(zip(*model_inds)),
                    self.dynamics_model.sy_train_in: inputs,
                    self.dynamics_model.sy_train_targ: targets
                }

                next_obs, _ = self.dynamics_model.sess.run(
                    (self._next_obs, self._adversarial_train_op),
                    feed_dict
                )

                obs = next_obs

                steps += 1
                if steps == self._epoch_length:
                    break

    def init_adversarial_model_update(self):
        """ Initialise update to adversarially modify the model.
        """

        # placeholders for adversarial update
        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self.dynamics_model._observation_shape),
            name='observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self.dynamics_model._action_shape),
            name='actions',
        )

        self._advantage_ph = tf.placeholder(
            tf.float32,
            shape=(None),
            name='advantage',
        )

        self._random_normal_ph = tf.placeholder(
            tf.float32,
            shape=(self.dynamics_model.num_nets, None, 1+self.dynamics_model._observation_shape)
        )

        self._model_inds_ph = tf.placeholder(
            tf.int32,
            shape=(None, 2)
        )

        inputs = tf.concat([self._observations_ph, self._actions_ph], -1)
        ensemble_model_means, ensemble_model_vars = self.dynamics_model._compile_outputs(inputs)
        batch_size = self._batch_size

        # because model predicts deltas for observations add original obs
        rewards_means = ensemble_model_means[:, :, 0:1]
        ensemble_model_means = tf.concat([rewards_means, ensemble_model_means[:,:,1:]+self._observations_ph], -1)
        ensemble_model_stds = tf.math.sqrt(ensemble_model_vars)
        ensemble_samples = tf.stop_gradient(ensemble_model_means + self._random_normal_ph * ensemble_model_stds)

        # use one model from ensemble
        model_inds = self._model_inds_ph
        samples = self._samples = tf.gather_nd(ensemble_samples, model_inds)
        self._model_stds = tf.gather_nd(ensemble_model_stds, model_inds)
        rewards = tf.squeeze(samples[:, :1])
        next_obs = self._next_obs = samples[:, 1:]

        # compute log probability of successor state
        def get_log_prob(states, means, stds):
            distribution = tfp.distributions.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds
            )
            state_log_prob = distribution.log_prob(states)[:]
            return state_log_prob

        # log prob for all ensemble members
        log_prob = self._log_prob = get_log_prob(
            samples,
            ensemble_model_means,
            ensemble_model_stds
        )

        # extract only the data from elites
        elite_inds = self.dynamics_model.get_elite_inds()
        log_prob = tf.gather(log_prob, elite_inds, axis=0)

        # correct for fact that transition is sampled uniformly from elites
        prob = tf.math.exp(tf.cast(log_prob, tf.float64))
        prob = prob * (1/len(elite_inds))
        prob = self._prob_corrected = tf.reduce_sum(prob, axis=0)
        log_prob = self._log_prob_corrected = tf.cast(tf.math.log(prob), tf.float32)

        adv_objective = self._advantage_ph * log_prob

        # total loss includes mle loss + lambda * adversarial loss
        supervised_loss = self.dynamics_model.train_loss
        total_loss = adv_objective * self._adversary_loss_weighting + supervised_loss

        self._adv_optimizer = tf.train.AdamOptimizer(learning_rate=self._adv_lr)
        self._adversarial_train_op = self._adv_optimizer.minimize(
            total_loss,
            var_list=self.dynamics_model.optvars
        )
        self.dynamics_model.sess.run(tf.variables_initializer(self._adv_optimizer.variables()))
