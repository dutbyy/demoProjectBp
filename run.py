import os


if __name__ == '__main__':
    from drill.bp.local.local import inference
    from config.config import DemoBuilder, agent_name_to_models
    inference(DemoBuilder, agent_name_to_models)
