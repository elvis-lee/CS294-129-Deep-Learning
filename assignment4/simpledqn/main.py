#!/usr/bin/env python
from collections import deque

import time
import chainer as C
import chainer.functions as F
import numpy as np
import pickle
import click
import gym
import torch
import torch.nn as nn
import os.path as osp

from simpledqn.replay_buffer import ReplayBuffer
import logger
from simpledqn.wrappers import NoopResetEnv, EpisodicLifeEnv

nprs = np.random.RandomState


def assert_allclose(a, b, atol=1e-6):
    if isinstance(a, (np.ndarray, float, int)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)
    elif isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list))
        assert len(a) == len(b)
        for a_i, b_i in zip(a, b):
            assert_allclose(a_i, b_i)
    elif isinstance(a, C.Variable):
        assert isinstance(b, C.Variable)
        assert_allclose(a.data, b.data)
    else:
        raise NotImplementedError


rng = nprs(42)


# ---------------------

class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.v = np.zeros(shape, dtype=np.float32)
        self.m = np.zeros(shape, dtype=np.float32)

    def step(self, g):
        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        step = - a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


# ---------------------

class NN(object):
    """Simple transparent neural network (multilayer perceptron) model.
    """

    def __init__(self, dims=None, out_fn=None):
        assert dims is not None
        assert out_fn is not None
        assert len(dims) >= 2

        self._out_fn = out_fn
        self.lst_w, self.lst_b = [], []
        for i in range(len(dims) - 1):
            shp = dims[i + 1], dims[i]
            # Correctly init weights.
            std = 0.01 if i == len(dims) - 2 else 1.0
            out = rng.randn(*shp).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            self.lst_w.append(C.Variable(out))
            self.lst_b.append(C.Variable(np.zeros(shp[0], dtype=np.float32)))
        self.train_vars = self.lst_w + self.lst_b

    def set_params(self, params):
        lst_wt, lst_bt = params
        for w, wt in zip(self.lst_w, lst_wt):
            w.data[...] = wt.data
        for b, bt in zip(self.lst_b, lst_bt):
            b.data[...] = bt.data

    def get_params(self):
        return self.lst_w, self.lst_b

    def dump(self, file_path=None):
        file = open(file_path, 'wb')
        pickle.dump(dict(w=self.lst_w, b=self.lst_b), file, -1)
        file.close()

    def load(self, file_path=None):
        file = open(file_path, 'rb')
        params = pickle.load(file)
        file.close()
        return params['w'], params['b']

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.lst_w, self.lst_b)):
            x = F.linear(x, w, b)
            if i != len(self.lst_w) - 1:
                x = F.tanh(x)
            else:
                return self._out_fn(x)


# ---------------------

def preprocess_obs_gridworld(obs):
    return obs.astype(np.float32)


def preprocess_obs_ram(obs):
    return obs.astype(np.float32) / 255.


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(1.0, float(t) / self.schedule_timesteps)
        return self.initial_p + (self.final_p - self.initial_p) * fraction


class NN_linear(nn.Module):
    def __init__(self, obs_size, act_size):
        super(NN_linear, self).__init__()
        self.Linear = nn.Linear(obs_size, act_size)

    def forward(self, obs):
        out = self.Linear(obs)
        return out

def test_loss(dqn, log_dir):
    test_args_reload = pickle.load(open(osp.join(log_dir, 'test_args.pkl'), 'rb'))
    nn_test = NN_linear(dqn._obs_dim, dqn._act_dim)
    nn_test_t = NN_linear(dqn._obs_dim, dqn._act_dim)
    nn_test.load_state_dict(torch.load(osp.join(log_dir, 'test_nn_linear.pkl')))
    nn_test_t.load_state_dict(torch.load(osp.join(log_dir, 'test_nn_t_linear.pkl')))
    old_q = dqn._q
    old_qt = dqn._qt
    dqn._q = nn_test
    dqn._qt = nn_test_t
    actual = dqn.compute_q_learning_loss(**test_args_reload).data.numpy()
    print(actual)
    tgt = np.array([2.21811724], dtype=np.float32)
    test_name = 'compute_q_learning_loss'
    try:
        assert_allclose(tgt, actual)
        print("Test for %s passed!" % test_name)
    except AssertionError as e:
        print(actual)
        print("Warning: test for %s didn't pass!" % test_name, "\nLoss should be: %f, yours is: %f" % (tgt[0], actual))
        raise e
    dqn._q = old_q
    dqn._qt = old_qt

if __name__ == "__main__":
    main()
