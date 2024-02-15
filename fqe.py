# +
import torch
import torch.nn as nn
import numpy as np

from math import ceil
from tqdm import trange

device = torch.device('cpu')

# +
def default_optimizer_factory(params):
    return torch.optim.AdamW(params, weight_decay=1e-4)

def get_optimizer(module_or_params, optimizer=None):
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    else:
        if optimizer is None:
            optimizer_factory = default_optimizer_factory
        else:
            assert callable(optimizer)
            optimizer_factory = optimizer
        parameters = module_or_params.parameters() if isinstance(module_or_params, nn.Module) else module_or_params
        return optimizer_factory(parameters)


# +
default_init_w = nn.init.xavier_normal_
default_init_b = nn.init.zeros_

def weight_initializer(init_w=default_init_w, init_b=default_init_b):
    def init_fn(m):
        if isinstance(m, nn.Linear):
            init_w(m.weight)
            init_b(m.bias)
    return init_fn

def dry_run(module, input_dim):
    """Just runs the network forward once and ignores errors.
    Seems to fix an uninformative PyTorch/CUDA error I was having, but not sure why."""
    try:
        with torch.no_grad():
            module(torchify(np.zeros((1, input_dim))))
    except:
        pass

def mlp(dims, layer_class=nn.Linear, activation=nn.ReLU(), output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(n_dims - 2):
        layers.append(layer_class(dims[i], dims[i+1]))
        layers.append(activation)
    layers.append(layer_class(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation)
    net = nn.Sequential(*layers)
    net.apply(weight_initializer())
    net.to(device=device, dtype=torch.float)
    dry_run(net, dims[0])
    return net


# -

class Module(nn.Module):
    def __call__(self, *args, **kwargs):
        args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in args]
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return super().__call__(*args, **kwargs)

    def save(self, f, prefix='', keep_vars=False):
        state_dict = self.state_dict(prefix=prefix, keep_vars=keep_vars)
        torch.save(f, state_dict)

    def load(self, f, map_location=None, strict=True):
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def try_load(self, f, **kwargs):
        try:
            self.load(f, **kwargs)
            return True
        except:
            return False


def torchify(x, double_to_float=True, int_to_long=True, to_device=True):
    if torch.is_tensor(x):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if x.dtype == torch.double:
        if double_to_float:
            x = x.float()
    elif x.dtype == torch.int:
        if int_to_long:
            x = x.long()

    if to_device:
        x = x

    return x


def batch_map(fn, args, batch_size=256, progress_bar=False):
    if type(args) in (list, tuple):
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert isinstance(arg_i, torch.Tensor)
            assert len(arg_i) == n
    else:
        n = len(args)
        args = [args]

    n_batches = ceil(float(n) / batch_size)
    
    iter_range_fn = trange if progress_bar else range
    results = []
    for batch_index in iter_range_fn(n_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), n)
        batch_output = fn(*[arg[batch_start:batch_end] for arg in args])
        results.append(batch_output)
    return torch.cat(results)


def epochal_training(compute_loss, optimizer, data, epochs, batch_size=256, max_grad_norm=None,
                     progress_bar=False):
    def one_step(batch):
        loss = compute_loss(*batch)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            for param_group in optimizer.param_groups:
                nn.utils.clip_grad_norm_(param_group['params'], max_grad_norm)
        optimizer.step()
        return loss.item()

    n = len(data[0])
    for i, data_i in enumerate(data):
        assert len(data_i) == n

    data = [torchify(data_i) for data_i in data]

    n_batches = ceil(float(n) / batch_size)
    iter_range_fn = trange if progress_bar else range
    losses = []
    for epoch_index in range(epochs):
        indices = torch.randperm(n)
        epoch_losses = []
        for batch_index in iter_range_fn(n_batches):
            batch_start = batch_size * batch_index
            batch_end = min(batch_size * (batch_index + 1), n)
            batch_indices = indices[batch_start:batch_end]
            loss_val = one_step([component[batch_indices] for component in data])
            epoch_losses.append(loss_val)
        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)
        
        print('Finished epoch {}/{}. Average loss: {}'.format(epoch_index+1, epochs, avg_epoch_loss))

    return losses

# +
class Critic(Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim+action_dim, *hidden_dim, 1])

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], 1))
    
class FQE:
    def __init__(self, policy, obs_dim, action_dim, result_path, discount=0.99, inner_epochs=20):
        self.policy = policy
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.result_path = result_path
        self.discount = discount
        self.inner_epochs = inner_epochs
        self.iterations_completed = 0
        self.criterion = nn.SmoothL1Loss()
        self.reset_critic()

    def reset_critic(self):
        self.critic = Critic(self.obs_dim, self.action_dim)
        self.critic_optimizer = get_optimizer(self.critic)

    def value(self, obs):
        with torch.no_grad():
            action = self.policy.sample_action(obs, deterministic=True)
            
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
                if len(obs.size())<2:
                    obs = torch.unsqueeze(obs, 0)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float)
                if len(action.size())<2:
                    action = torch.unsqueeze(action, 0)
                    
            values = self.critic(obs, action)
            return values

    def compute_critic_target(self, reward, not_done, next_obs):
        target = reward + not_done * self.discount * self.value(next_obs) 
        return target

    def compute_critic_loss(self, obs, action, target):
        return self.criterion(self.critic(obs, action), target)

    def train(self, replay_buffer, iterations=1, map_batch_size=1000):
        data = replay_buffer.sample(500000)
        obs, action, next_obs, done, reward = torch.from_numpy(data["observations"]), torch.from_numpy(data["actions"]), \
            torch.from_numpy(data["next_observations"]), torch.from_numpy(data["terminals"]), torch.from_numpy(data["rewards"])
        
        not_done = 1 - done
        
        for _ in range(iterations):
            target = batch_map(self.compute_critic_target, [reward, not_done, next_obs], batch_size=map_batch_size)
            
            epochal_training(self.compute_critic_loss, self.critic_optimizer, [obs, action, target],
                             epochs=self.inner_epochs, batch_size=map_batch_size)

            self.iterations_completed += 1
        
    def load(self, name):
        self.critic.load_state_dict(torch.load(name))
