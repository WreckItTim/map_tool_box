from map_tool_box.modules import Component

class Model(Component.Component):
    pass

def read_sb3_model(model_path, device='cuda'):
    from map_tool_box.modules import SB3Wrapper
    from stable_baselines3 import DQN
    model = DQN.load(model_path, device=device)
    model = SB3Wrapper.ModelSB3(model)
    return model

def read_model(model_path, device='cuda'):
    if '.zip' in str(model_path):
        return read_sb3_model(model_path, device=device)