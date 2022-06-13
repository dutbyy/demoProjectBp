from drill.lib.feature.feature_set import CommonFeatureSet
from drill.lib.feature import EntityFeatureSet, CommonFeatureSet
from drill.lib.feature import PlainFeature, VectorFeature, OnehotFeature, RangedFeature
from drill.lib.network.encoder import EntityEncoder, CommonEncoder
from drill.lib.network.decoder import CategoricalDecoder, UnorderedMultiSelectiveDecoder, SingleSelectiveDecoder
from drill.lib.network.layer import ValueApproximator
from drill.lib.action_head import ActionHead
from drill.lib.network.commander import CommanderNetwork
from drill.lib.network.aggregator import GRUAggregator
from drill.bp.builder import Builder
from drill.bp.commander_agent import CommanderAgent


from interface.pipeline import DemoPipeline
from interface.env import DemoEnv


# define the commander network
common_feature_set = CommonFeatureSet(
    name="common",
    feature_dict={
        'plain':  PlainFeature(),
        'vector': VectorFeature(length=2),
        'ranged': RangedFeature(low=0.0, high=100.0, length=1),
        'onehot': OnehotFeature(depth=10),
    }
)
entity_feature_set = EntityFeatureSet(
    name = 'entity',
    max_length = 10,
    feature_dict = {
        'plain': PlainFeature(),
    }
)

common_encoder = CommonEncoder(hidden_layer_sizes=[256, 128])
entity_encoder = EntityEncoder(hidden_layer_sizes=[256, 128])
feature2encoder = {
    common_feature_set.name: common_encoder,
    entity_feature_set.name: entity_encoder,
}
aggregator = GRUAggregator(hidden_layer_sizes=[512, 256],
                           state_size=256,
                           output_size=512)

course_decoder = CategoricalDecoder(36, hidden_layer_sizes=[512, 256])
course_action_head = ActionHead("course")
action2decoder = {
    course_action_head: course_decoder
}
value_approximator = ValueApproximator(hidden_layer_sizes=[64, 32])
commander_network = CommanderNetwork(feature2encoder, aggregator, action2decoder, value_approximator)

agent_name_to_models = {'demo': 'demo_model'}

EnvConfig = {
}

# define the builder
class DemoBuilder(Builder):

    @staticmethod
    def get_model_name(agent_name):
        return agent_name_to_models[agent_name]

    @staticmethod
    def build_env(env_id):
        EnvConfig['env_id'] = env_id
        return DemoEnv(EnvConfig)

    @staticmethod
    def build_agent(model_name):
        print("build agent: model name is :", model_name)
        pipeline: DemoPipeline = DemoBuilder.build_pipeline()
        fake_inputs = pipeline.get_fake_state(model_name, batch_size=1)
        commander_network(fake_inputs)
        agent = CommanderAgent(model_name, commander_network)
        return agent

    @staticmethod
    def build_pipeline():
        feature_sets = [ common_feature_set, entity_feature_set ]
        action_heads = [ course_action_head ]
        return DemoPipeline({'demo_model': feature_sets},{'demo_model': action_heads}, agent_name_to_models)

    @staticmethod
    def build_fake_inputs(model_name):
        pipeline: DemoPipeline = DemoBuilder.build_pipeline()
        fake_inputs_dict = pipeline.get_fake_state(model_name)
        fake_inputs_dict["reward"] = np.array(0, np.float32)
        return fake_inputs_dict

flow_no_resource_conf = {
    'log_dir': '/job/logs/logs',
    'Actor': {'envs_per_worker': 1},
    'Predictor': {'cpus_per_gpu_worker': 8},
    'Learner': {
        'players': ['player0'],
        'cpus_per_worker': 3,
        'sample_mode': 'LIFO',  # currently only support this mode
        'fragment_size': 128,
        'replay_size': 128,
        'sample_batch_size': 2048,  # 2048*16
        'putback_replays': False,
        #'checkpoint_dir': '/job/model',
        'checkpoint_dir': './appo_checkpoint',
        'checkpoint_weight_version_interval': 10000,
        'max_data_reuse': 5,
        # gpu batch size in corresponding predictor service id
        'predictor_gpu_batch_size': 170,
        # it works if there is at least a cpu worker for predictor
        'predictor_cpu_batch_size': 64,
    }
}
