#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# This code is modified to test the SALINA a2c algorithm
# on GPU using a toy dataset (FOREX_EURUSD_1H_ASK) for financial time series.
# Francesco Bardozzo, PhD - NeuroneLab - University of Salerno (IT)

import copy
import time
import numpy as np
import gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import TAgent, Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger

import tensorflow as tf


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v

def _gen_state(observation):
    # observation = observation[np.newaxis, ...] no
    index = torch.tensor([0])
    diff_close = torch.transpose(torch.index_select(observation[0], 1, index), 1, 0)
    index2 = torch.tensor([1])
    buy_sell = torch.transpose(torch.index_select(observation[0], 1, index2), 1, 0)
    observation = torch.squeeze(torch.stack([diff_close, buy_sell], dim=1))
    return  observation

def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


class ProbAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__(name="prob_agent")
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(_gen_state(observation))
        #scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)


class ActionAgent(TAgent):
    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class CriticAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.critic_model(_gen_state(observation)).squeeze(-1)
        self.set(("critic", t), critic)

def stock_func(max_episode_steps,seed=123, window_size =10, size_sample=100):
    #for now doesn't work...
    df =  FOREX_EURUSD_1H_ASK.copy()
    start_index = window_size
    if size_sample < 0:
        end_index = len(df)
    else:
        end_index = size_sample

    env = TimeLimit(gym.make('forex-v0', df=df, window_size=window_size, frame_bound=(start_index, end_index)), max_episode_steps=max_episode_steps)

    #env = TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)
    env.seed(seed)
    return env


def run_a2c(cfg):
    logger = instantiate_class(cfg.logger)

    env = instantiate_class(cfg.algorithm.env)
    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    #del env

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    acq_env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )
    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )
    acq_prob_agent = copy.deepcopy(prob_agent)
    acq_action_agent = ActionAgent()
    acq_agent = TemporalAgent(Agents(acq_env_agent, acq_prob_agent, acq_action_agent))
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
    )
    acq_remote_agent.seed(cfg.algorithm.env_seed)

    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )

    tprob_agent = TemporalAgent(prob_agent).to(device=cfg.algorithm.device)
    tcritic_agent = TemporalAgent(critic_agent).to(device=cfg.algorithm.device)

    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)

    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        for a in acq_remote_agent.get_by_name("prob_agent"):
            a.load_state_dict(prob_agent.state_dict())

        if epoch > 0:
            acq_workspace.copy_n_last_steps(1)
            acq_remote_agent(
                acq_workspace,
                t=1,
                n_steps=cfg.algorithm.n_timesteps - 1,
                stochastic=True,
            )
        else:
            acq_remote_agent(
                acq_workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True
            )

        replay_workspace = Workspace(acq_workspace).to(device=cfg.algorithm.device)
        tprob_agent(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)
        tcritic_agent(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)

        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]

        td_error = td ** 2
        critic_loss = td_error.mean()

        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        logger.add_scalar("critic_loss", critic_loss.item(), epoch)
        logger.add_scalar("entropy_loss", entropy_loss.item(), epoch)
        logger.add_scalar("a2c_loss", a2c_loss.item(), epoch)

        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        creward = replay_workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("reward", creward.mean().item(), epoch)



@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_a2c(cfg)


if __name__ == "__main__":
    main()
