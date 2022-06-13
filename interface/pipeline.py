from drill.bp.rl_pipeline import RLPipeline
from drill.bp.common import ACTION_PREFIX
from drill.lib.utils import feature_sets_process
import numpy as np


class DemoPipeline(RLPipeline):

    def __init__(self, feature_sets_dict, action_heads_dict, agent2model, gamma=0.99, lamb=0.95):
        super().__init__(feature_sets_dict,action_heads_dict,gamma=gamma,lamb=lamb)
        self.agent2model = agent2model

    def get_fake_state(self, model_name, batch_size=None):
        print(model_name)
        print(self._feature_sets_dict)
        feature_sets = self._feature_sets_dict[model_name]
        fake_state_dict = {}
        for feature_set in feature_sets:
            if batch_size is None:
                feature = np.random.randn(*feature_set.shape).astype(np.float32)
            else:
                feature = np.random.randn(batch_size, *feature_set.shape).astype(np.float32)
            fake_state_dict[feature_set.name] = feature
        return fake_state_dict

    def obs2state(self, obs, last_predict_output_dict):
        name2feature = {
            'common': {
                'plain': 3.1,
                'vector' : [1, 2],
                'ranged' : 69,
                'onehot': 3,
            },
            'entity': [
                {
                    'plain': 3.1,
                },
                {
                    'plain': 3.1,
                }
            ]
        }
        state_dict = {}
        for agent_name, model_name in self.agent2model.items():
            feature_sets = self._feature_sets_dict[model_name]
            state_dict[agent_name] = feature_sets_process(
                feature_sets, name2feature)
        return state_dict

    def reward(self, obs, done, extra_info_dict):
        reward = 0
        reward_dict = {}
        for agent_name in self.agent2model.keys():
            reward_dict[agent_name] = reward
        return reward_dict

    def action2command(self, agent_name, action_dict, obs):
        cmds = {}
        return cmds

    def valid_action(self, agent_name, action_dict):
        valid_action_dict = {}
        for k, v in action_dict.items():
            valid_action_dict[k] = True
        return valid_action_dict

