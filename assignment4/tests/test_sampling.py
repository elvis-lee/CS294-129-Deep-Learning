import random
import unittest
from utils import EnvPool, parallel_collect_samples
import gym


class DummyEnv(gym.Env):
    def _reset(self):
        return 0

    def _step(self, action):
        return random.choice(range(10)), 1., random.choice([False] * 9 + [True]), dict()


class DummyEnvMaker(object):
    def __init__(self):
        pass

    def make(self):
        return DummyEnv()


class DummyPolicy(object):
    def get_actions(self, obs):
        return [random.choice(range(10)) for _ in obs]


class TestSampling(unittest.TestCase):
    def test_env_pool(self):
        with EnvPool(env_maker=DummyEnvMaker(), n_envs=10, n_parallel=3) as pool:
            obs = pool.reset()
            self.assertEqual(len(obs), 10)

    def test_parallel_sampler(self):
        with EnvPool(env_maker=DummyEnvMaker(), n_envs=10, n_parallel=3) as env_pool:
            trajs = parallel_collect_samples(env_pool, DummyPolicy(), num_samples=1000)
            self.assertEqual(sum([len(traj['rewards']) for traj in trajs]), 1000)
        with EnvPool(env_maker=DummyEnvMaker(), n_envs=10, n_parallel=3) as env_pool:
            trajs = parallel_collect_samples(env_pool, DummyPolicy(), num_samples=1005)
            self.assertEqual(sum([len(traj['rewards']) for traj in trajs]), 1010)


if __name__ == "__main__":
    unittest.main()
