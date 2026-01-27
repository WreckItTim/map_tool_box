from map_tool_box.modules import Component
from map_tool_box.modules import Control
from map_tool_box.modules import Utils
from pathlib import Path
import os

# used to evaluate at end of episode
class Evaluator(Component.Component):
    
    # how do we evalute the given environment?
    def evaluate(self, environment):
        raise NotImplementedError

class Curriculum(Evaluator):
    def __init__(self, difficulties, level_up_freq, burn_in):
        self.difficulties = difficulties
        self.difficulty_index = 0
        self.level_up_freq = level_up_freq
        self.burn_in = burn_in

    def level_up(self):
        self.difficulty_index += 1
        if self.difficulty_index < len(self.difficulties):
            difficulty = self.difficulties[self.difficulty_index]
        else:
            difficulty = 'finished'
        return difficulty

    def end(self, environment):
        n_episodes = environment.get_episodes()
        if n_episodes >= self.burn_in and n_episodes % self.level_up_freq == 0:
            difficulty = self.level_up()
            environment.set_difficulty(difficulty)

class LearningCurve(Evaluator):
    
    def __init__(self, evaluation_environments, model, eval_frequency, write_directory, print_progress=True):
        self.evaluation_environments = evaluation_environments
        self.model = model
        self.eval_frequency = eval_frequency
        self.print_progress = print_progress
        os.makedirs(write_directory, exist_ok=True)
        self.write_path = Path(write_directory, 'learning_curve.p')
        self.curves = {key:[] for key in evaluation_environments}

    def eval(self):
        for key in self.evaluation_environments:
            environment = self.evaluation_environments[key]
            episodes_states, accuracy = Control.eval(environment, self.model)
            self.curves[key].append(accuracy)
        Utils.pickle_write(self.write_path, self.curves)
        if self.print_progress:
            print(self.n_episodes, [f'{key}: {self.curves[key][-1]}' for key in self.curves])

    def end(self, environment):
        self.n_episodes = environment.get_episodes()
        if self.n_episodes % self.eval_frequency == 0:
            self.eval()

class MaxEpisodes(Evaluator):
    
    def __init__(self, max_episodes):
        self.max_episodes = max_episodes

    def end(self, environment):
        n_episodes = environment.get_episodes()
        if n_episodes >= self.max_episodes:
            environment.stop_learning()
            