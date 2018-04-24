from algs import pg
from env_makers import EnvMaker
from models import CategoricalMLPPolicy, MLPValueFunction
from cloudexec import *
import logger
import cloudpickle
import os


class SnapshotSaver(object):
    def __init__(self, dir):
        self.dir = dir

    def save_state(self, state):
        with open(os.path.join(self.dir, "snapshot.pkl"), "wb") as f:
            cloudpickle.dump(state, f)

    def get_state(self):
        with open(os.path.join(self.dir, "snapshot.pkl"), "rb") as f:
            return cloudpickle.load(f)


def main(*_):
    saver = SnapshotSaver(logger.get_dir())
    state = saver.get_state()
    env = state['alg_state']['env_maker'].make()

    alg = state['alg']
    alg(env=env, snapshot_saver=saver, **state['alg_state'])
    # import ipdb; ipdb.set_trace()
    #
    # env_maker = EnvMaker('CartPole-v0')
    # env = env_maker.make()
    # policy = CategoricalMLPPolicy(env.observation_space, env.action_space, env.spec)
    # baseline = MLPValueFunction(env.observation_space, env.action_space, env.spec)
    #
    # pg(
    #     env=env,
    #     env_maker=env_maker,
    #     policy=policy,
    #     baseline=baseline,
    #     snapshot_saver=saver,
    # )


if __name__ == "__main__":
    remote_call(
        Task(main, dict()),
        Config(exp_group="test-resume", exp_name="test-resume"),
        mode=local_mode
    )
