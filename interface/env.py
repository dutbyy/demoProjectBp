import time
from drill.bp.env import Env

class Environment:
    def __init__(self, config):
        pass
    def reset(self):
        pass

    def get_obs(self):
        return {'abc': 1}, {}

    def step(self, cmds):
        pass

class DemoEnv(Env):
    def __init__(self, config):
        env_id = config['env_id']
        self.done = False
        self.error = False
        self.steps = 1
        self._env = Environment({})
        print('Environment is ready')

    def reset(self):
        #print('reset the Env')
        self._env.reset()
        time.sleep(.1)
        self.done = False
        self.error = False
        self.steps = 1
        raw_obs, info = self._env.get_obs()
        return raw_obs

    def step(self, command_dict):
        self.steps+=1
        self._env.step(command_dict)
        if self.steps >= 100:
            self.done = True
        raw_obs, info = self._env.get_obs()
        return raw_obs, self.done, info

