from map_tool_box.modules import Component
import numpy as np
import random

# gets starting position and possibly target
class Spawner(Component.Component):
    
    def spawn(self):
        raise NotImplementedError

# helper funcs
def set_environment_vars(environment, path):
    start = path[0]
    target = path[-1]
    target.direction = None
    environment.set_point(start)
    environment.set_target(target)
    environment.set_path(path)

# randomly selects from a set of paths, given the current difficulty
class CurricululmTrain(Spawner):

    # paths_dict is dictionary {difficulty:paths} {str:list}
        # NOTE: key order must be sorted from lowest to highest difficulty
    # samples a path at given difficulty
    # lower_difficulty_proba is probability to sample from a lower difficulty
    def __init__(self, paths_dict, lower_difficulty_proba=0.3):
        self.paths_dict = paths_dict
        self.lower_difficulty_proba = lower_difficulty_proba
        self.difficulties = list(paths_dict.keys())

    def get_selectable_difficulties(self, max_difficulty):
        max_idx = self.difficulties.index(max_difficulty)
        difficulties = [difficulty for difficulty in self.difficulties[:max_idx]]
        return difficulties

    def start(self, environment):
        difficulty = environment.get_difficulty()
        path = self.spawn(difficulty)
        set_environment_vars(environment, path)
        
    # randomly fetch next path
    def spawn(self, difficulty):
        if difficulty == 'finished':
            difficulty = random.choice(self.difficulties)
        elif self.lower_difficulty_proba > 0 and random.random() < self.lower_difficulty_proba:
            selectable_difficulties = self.get_selectable_difficulties(difficulty)
            if len(selectable_difficulties) > 0:
                difficulty = random.choice(selectable_difficulties)
        paths = self.paths_dict[difficulty]
        path = random.choice(paths)
        return path

# iteratively gets one path after the other
class CurricululmEval(Spawner):

    # paths_dict is dictionary {difficulty:paths} {str:list}
        # NOTE: key order must be sorted from lowest to highest difficulty
    # samples a path at given difficulty
    # lower_difficulty_proba is probability to sample from a lower difficulty
    def __init__(self, paths_dict):
        self.paths_dict = paths_dict
        self.difficulties = list(paths_dict.keys())
        self.reset()

    def reset(self):
        self.path_index = 0
        self.difficulty_index = 0
        self.difficulty = self.difficulties[self.difficulty_index]
        self.has_more = True

    def has_more_paths(self):
        return self.has_more

    def start(self, environment):
        path = self.spawn()
        set_environment_vars(environment, path)

    # iterate to next path, and potentially next difficulty
    def iterate(self):
        self.path_index += 1
        if self.path_index >= len(self.paths_dict[self.difficulty]):
            self.path_index = 0
            self.difficulty_index += 1
            if self.difficulty_index >= len(self.difficulties):
                self.has_more = False
            else:
                self.difficulty = self.difficulties[self.difficulty_index]
        
    # randomly fetch next path
    def spawn(self):
        paths = self.paths_dict[self.difficulty]
        path = paths[self.path_index]
        self.iterate()
        return path