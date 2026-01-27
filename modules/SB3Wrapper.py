from map_tool_box.modules import Environment
from map_tool_box.modules import Observer
from map_tool_box.modules import Actor
import numpy as np
import gymnasium

# to use the third party library StableBaselines3 (SB3) model use the following wrappers

class SB3Environment(gymnasium.Env, Environment.ReinforcementLearning):

    # environment is one of class objects defined above
    def __init__(self, environment, observer_sb3, actor_sb3):
        object.__setattr__(self, 'environment', environment)
        # even though we do not directly use the observation or action space, these fields are necesary for sb3
        self.observation_space = observer_sb3.get_space()
        self.action_space = actor_sb3.get_space()

    # get original object attributes    
    def __getattr__(self, name):
        return getattr(self.environment, name)
    # set original object attributes    
    def __setattr__(self, name, value):
        setattr(self.environment, name, value)
    # delete original object attributes    
    def __delattr__(self, name):
        delattr(self.environment, name)
    
    def step(self, action_value):
        observation, reward, terminate, state = self.environment.step(action_value)
        truncate = False
        if terminate:
            self.environment.end()
            terminator_reason = state['end']
            if terminator_reason not in ['Goal']:
                truncate = True
        info = {}
        return observation, reward, terminate, truncate, info
        
    def reset(self, seed=None, options=None):
        observation = self.environment.start()
        info = {}
        return observation

def get_array_space(observer):
    return gymnasium.spaces.Box(low=0, high=255, shape=observer.history_shape, dtype=np.uint8)

def get_vector_space(observer):
    return gymnasium.spaces.Box(0, 1, shape=observer.history_shape, dtype=np.float32)

class SB3Observer(Observer.Observer):
    
    def __init__(self, observer):
        object.__setattr__(self, 'observer', observer)

    # get original object attributes    
    def __getattr__(self, name):
        return getattr(self.observer, name)
    # set original object attributes    
    def __setattr__(self, name, value):
        setattr(self.observer, name, value)
    # delete original object attributes    
    def __delattr__(self, name):
        delattr(self.observer, name)
    
    def get_space(self):
        gym_space = None
        if self.observer.o_type == 'dict':
            spaces = {}
            for key in self.observer.observer_dict:
                observer = self.observer.observer_dict[key]
                if observer.o_type == 'array':
                    spaces[key] = get_array_space(observer)
                if observer.o_type == 'vector':
                    spaces[key] = get_vector_space(observer)
            gym_space = gymnasium.spaces.Dict(spaces=spaces)
        elif self.observer.o_type == 'array':
            gym_space = get_array_space(self.observer)
        elif self.observer.o_type == 'vector':
            gym_space = get_vector_space(self.observer)
        return gym_space

class SB3Actor(Actor.Actor):
    
    def __init__(self, actor):
        object.__setattr__(self, 'actor', actor)

    # get original object attributes    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    # set original object attributes    
    def __setattr__(self, name, value):
        setattr(self.actor, name, value)
    # delete original object attributes    
    def __delattr__(self, name):
        delattr(self.actor, name)
    
    def get_space(self):
        return gymnasium.spaces.Discrete(len(self.actor.actions))
        # return gymnasium.spaces.Box(min_space, max_space) # continuous action space TODO

# takes sb3 model as input (reverse wrap)
class ModelSB3:

    def __init__(self, sb3model):
        object.__setattr__(self, 'sb3model', sb3model)

    # get original object attributes    
    def __getattr__(self, name):
        return getattr(self.sb3model, name)
    # set original object attributes    
    def __setattr__(self, name, value):
        setattr(self.sb3model, name, value)
    # delete original object attributes    
    def __delattr__(self, name):
        delattr(self.sb3model, name)

    def predict(self, observations):
        action, _states = self.sb3model.predict(observations, deterministic=True)
        return action