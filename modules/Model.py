
def read_sb3_model(model_path):
    from map_tool_box.modules import SB3Wrapper
    from stable_baselines3 import DQN
    model = DQN.load(model_path)
    model = SB3Wrapper.ModelSB3(model)
    return model

def read_model(model_path):
    if '.zip' in str(model_path):
        return read_sb3_model(model_path)