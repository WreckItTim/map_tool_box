from map_tool_box.modules import Component
import numpy as np
import math

# used to determine if we terminate an episode
class Terminator(Component.Component):
    
    # how do we check for termination?
    def check(self, environment):
        raise NotImplementedError

# updates state dictionary with terminate value if the maximum number of steps is reached
# steps_multiplier is used to determine max steps based on expected steps for current episode
class MaxSteps(Terminator):
    
    def __init__(self, steps_multiplier):
        self.steps_multiplier = steps_multiplier
        
    def start(self, environment):
        path_steps = environment.get_path_steps()
        self.max_steps = math.ceil(self.steps_multiplier * path_steps)

    def step(self, environment):
        n_steps = environment.get_steps()
        terminate = self.check(n_steps)
        return terminate
        
    def check(self, n_steps):
        terminate = n_steps >= self.max_steps
        return terminate

# updates state dictionary with terminate value if reached goal within threshold
class Goal(Terminator):
    
    def __init__(self, goal_tolerance):
        self.goal_tolerance = goal_tolerance
        
    def step(self, environment):
        goal_distance = environment.get_goal_distance()
        terminate = self.check(goal_distance)
        return terminate
        
    def check(self, goal_distance):
        terminate = goal_distance <= self.goal_tolerance
        return terminate