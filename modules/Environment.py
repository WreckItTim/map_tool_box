from matplotlib.animation import FuncAnimation
from map_tool_box.modules import Utils
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import copy

        
# define an environment which we can move an agent through
# considers episodic movement that progresses one step at a time until termination criteria is met
# one episode uses the class methods: start(), step(), ..., step(), end()
# this is heavily inspired by OpenAI Gymnaisum environment objects
    # see ReinforcementLearning class below and SB3Wrappers.py module to be fully compatible
# self.calculations={} contains key-value pairs of intermediate calculations
    # this can be used to avoid reduntant calculations for efficiency
class Episodic:
    # Component parameters have the parent-inherited methods: start(), step(), and end() which take environment as input
    # spawner, actions, observer, terminators, and others are components
        # spawner.start() sets initial environment x, y, z, direction, (target) values 
        # actor.step() updates intermediate environment x, y, z, direction values
        # observer.step() returns a processed observation captured at current state
        # terminate.step() returns a boolean determining if we should end the episode
        # other is used to trigger miscellaneous functionality on start/step/end as user defined
    # grid object uses geo x,y,z,direction coordinates to determine collisions and out-of-bounds
        # if grid is a Data_Map object then this is used to access cached observations at given coordinates as well
    # ckpt_freq=x where x is the frequency of episodes to call checkpoint_out() on each component
    def __init__(self, grid, spawner, actor, observer, terminators, others=[], 
                 ckpt_freq=None, ckpt_dir=None):
        self.grid = grid
        self.spawner = spawner
        self.actor = actor
        self.observer = observer
        self.terminators = terminators
        self.others = others
        self.n_episodes = 0
        self.ckpt_freq = ckpt_freq
        if ckpt_freq is not None and ckpt_dir is None:
            print('ERROR: ckpt_freq set but no ckpt_dir set to write checkpoints to')
        self.ckpt_dir = ckpt_dir
        self.other_ckpt_objs = []

    # add duck class object to checkpoint
    def add_checkpoint_obj(self, obj):
        self.other_ckpt_objs.append(obj)

    # checkpoint handling (to continue from another process)
    def checkpoint_in(self, ckpt_dir):
        # read ckpt params for this environment
        self.set_difficulty(Utils.pickle_read(Path(ckpt_dir, 'env__difficulty.p')))
        self.n_episodes = Utils.pickle_read(Path(ckpt_dir, 'env__n_episodes.p'))
        self.continue_learning = Utils.pickle_read(Path(ckpt_dir, 'env__continue_learning.p'))
        # read ckpt params for all components
        self.spawner.checkpoint_in(ckpt_dir)
        self.actor.checkpoint_in(ckpt_dir)
        self.observer.checkpoint_in(ckpt_dir)
        for terminator in self.terminators:
            terminator.checkpoint_in(ckpt_dir)
        for other in self.others:
            other.checkpoint_in(ckpt_dir)
        for other in self.other_ckpt_objs:
            other.checkpoint_in(ckpt_dir)
    def checkpoint_out(self):
        # write ckpt params from this environment
        Utils.pickle_write(Path(self.ckpt_dir, 'env__difficulty.p'), self.get_difficulty())
        Utils.pickle_write(Path(self.ckpt_dir, 'env__n_episodes.p'), self.n_episodes)
        Utils.pickle_write(Path(self.ckpt_dir, 'env__continue_learning.p'), self.continue_learning)
        # write ckpt params from all components
        self.spawner.checkpoint_out(self.ckpt_dir)
        self.actor.checkpoint_out(self.ckpt_dir)
        self.observer.checkpoint_out(self.ckpt_dir)
        for terminator in self.terminators:
            terminator.checkpoint_out(self.ckpt_dir)
        for other in self.others:
            other.checkpoint_out(self.ckpt_dir)
        for other in self.other_ckpt_objs:
            other.checkpoint_out(self.ckpt_dir)

    
    #### GETTERS AND SETTERS
    
    # returns translational and rotational pose of agent and goal on grid
    def get_point(self):
        return self.point
    def set_point(self, point):
        self.point = point
        self.set_position(np.array([point.x, point.y, point.z]))
    def get_target(self):
        return self.target
    def set_target(self, target):
        self.target = target
        self.set_goal(np.array([target.x, target.y, target.z]))
    def set_position(self, position):
        self.position = position
    def get_position(self):
        return self.position   
    def set_goal(self, goal):
        self.goal = goal
    def get_goal(self):
        return self.goal

    # returns 3d occupancy grid 
    def get_grid(self):
        return self.grid

    # get number of steps taken this episode
    def get_steps(self):
        return self.n_steps

    # get number of episodes taken by this environment
    def get_episodes(self):
        return self.n_episodes

    # get the most recent action value passed into step()
    def get_action_value(self):
        return self.action_value

    # this func assumes path is set (by spawner)
    def get_path(self):
        return self.path   
    def set_path(self, path):
        self.path = path
    # returns expected number of steps for current path
    def get_path_steps(self):
        path_steps = len(self.path)-1
        return path_steps

    # difficulty is used for curriculum learning to determine spawns
    def get_difficulty(self):
        return self.difficulty
    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        
    # check if calculation has been made, otherwise get Euclidean displacement
    def get_goal_displacement(self):
        if 'goal_displacement' not in self.calculations:
            self.calculations['goal_displacement'] = self.goal - self.position
        return self.calculations['goal_displacement']

    # check if calculation has been made, otherwise get Euclidean distance
    def get_goal_distance(self):
        if 'goal_distance' not in self.calculations:
            goal_displacement = self.get_goal_displacement()
            self.calculations['goal_distance'] = np.linalg.norm(goal_displacement)
        return self.calculations['goal_distance']

    
    #### EPISODIC CONTROL
    
    # beginning of a new episode
    def start(self):
        
        # reset episodic state variables
        self.n_steps = 0
        self.calculations = {}
        
        # start all components
        self.spawner.start(self) # sets x, y, z, direction, (path)
        self.initial_target = self.target
        self.actor.start(self)
        self.observer.start(self)
        for terminator in self.terminators:
            terminator.start(self)
            
        # observation space
        observations = copy.deepcopy(self.observer.step(self))
        self.observations = observations
        
        for other in self.others:
            other.start(self)

        # initialize path history
        self.path_history = [self.point]

        # initial state variables
        self.state = {}
        self.state['step'] = 0
        self.state['action_value'] = 'start'
        self.state['point'] = self.point
        self.state['target'] = self.target
        self.state['initial_target'] = self.initial_target
        self.state['path_steps'] = self.get_path_steps()
            
        return observations, self.state
        
    # intermediate steps of episode
    # action_value is used to determine what actions to take -- as passed into the self.actor object
        # this is intentially arbitrary as different actor-models take different action_values as input
    def step(self, action_value):
        
        # reset calculations
        self.calculations = {} 
        self.n_steps += 1
        
        # action space
        self.action_value = action_value # save for future access
        self.actor.step(self) # sets point at each action.step()

        # add to path history
        self.path_history.append(self.point)

        # check if we should terminate episode
        self.terminate = False
        self.terminator_reason = None
        for terminator in self.terminators:
            self.terminate = terminator.step(self)
            if self.terminate:
                self.terminator_reason = terminator.__class__.__name__
                break

        # call step for any other components
        for other in self.others:
            other.step(self)
            
        # observation space
        observations = copy.deepcopy(self.observer.step(self))
        self.observations = observations

        # track state variables
        self.state = {}
        self.state['step'] = self.n_steps
        self.state['action_value'] = action_value
        self.state['point'] = self.point # set by actor
        self.state['target'] = self.target # set by spawner/other
        if self.terminate:
            self.state['end'] = self.terminator_reason
        
        return observations, self.terminate, self.state

    # end of epsiode
    def end(self):
        
        self.n_episodes += 1
        
        # end all components
        self.spawner.end(self)
        self.actor.end(self)
        self.observer.end(self)
        for terminator in self.terminators:
            terminator.end(self)
        for other in self.others:
            other.end(self)

        # checkpoint?
        if self.ckpt_freq is not None:
            if self.n_episodes % self.ckpt_freq == 0:
                self.checkpoint_out()

    # this is a slightly hard-coded method used for debugging
    # returns an animation of an episode based on given list of states
    def animate_episode(self, states, show_rewards=True):

        # get some params
        last_point = states[0]['point']
        path_steps = states[0]['path_steps']
        initial_target = states[0]['initial_target']
        n_history = self.observer.observer_dict['img'].n_history
        n_elements = self.observer.observer_dict['vec'].n_elements
        N_frames = len(states)

        # initialize pyplot 
        ncols = 3
        line_height = 50 # # pixels for height of one line of text
        x_offset = 0 # text offset from left-x
        fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(1+(n_history/ncols)))
        ax = axs[0,0]
        self.grid.plot_map(fig, ax, show_z=False)
        ax.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0)

        # call this each frame
        def update(t):
            state = states[t]
            step = state['step']
            point = state['point']
            target_point = state['target']
            last_point = None
            if t > 0:
                last_point = states[t-1]['point']
            action_value = state['action_value']
            observations = state['observations']
            
            # plot agent on map
            ax = axs[0,0]
            ax.clear()
            ax.axis('off')
            ax.set_title(f'Birds-Eye-View')
            self.grid.plot_map(fig, ax, show_z=False)
            ax.scatter(initial_target.x+self.grid.x_shift, initial_target.y+self.grid.y_shift, color='green', 
                       marker='*', s=128)
            ax.scatter(target_point.x+self.grid.x_shift, target_point.y+self.grid.y_shift, color='green', 
                       marker='.', s=64)
            self.grid.plot_agent(fig, ax, point)
            
            # add first text (position, target, actions, vector-observations) to panel
            ax = axs[0,1]
            ax.clear()
            ax.axis('off')
            ax.set_title(f'State')
            ax.imshow(np.full([480, 480], 255).astype(np.uint8), vmin=0, vmax=255, cmap='grey')
            text_lines = []
            text_lines.append(f'step #: {step}')
            if last_point is None:
                text_lines.append('')
            else:
                text_lines.append(f'{last_point}')
            if isinstance(action_value, str):
                text_lines.append(f'action: {action_value}')
            else:
                text_lines.append(f'action: {self.actor.actions[action_value]}')
            text_lines.append(f'{point}')
            text_lines.append(f'waypoint: {target_point}')
            text_lines.append(f'target: {initial_target}')
            text_lines.append(f'optimal # steps: {path_steps}')
            for t in range(n_history):
                obs = observations['vec'][t*n_elements:(t+1)*n_elements]
                line = f'vec t-{t}:'
                for value in obs:
                    line += f' {value:.2f}'
                text_lines.append(line)
            for line_idx, text_line in enumerate(text_lines):
                ax.text(x_offset, line_height+line_height*line_idx, text_line)
            
            # add second text (rewards) to panel
            ax = axs[0,2]
            ax.clear()
            ax.axis('off')
            text_lines = []
            ax.imshow(np.full([480, 480], 255).astype(np.uint8), vmin=0, vmax=255, cmap='grey')
            if show_rewards:
                ax.set_title(f'Rewards')
                if 'reward_dict' in state:
                    reward_dict = state['reward_dict'].copy()
                    for key in reward_dict:
                        text_lines.append(f'{key}: {reward_dict[key]:.2f}')
            if 'end' in state:
                end = state['end']
                text_lines.append(f'end: {end}')
            for line_idx, text_line in enumerate(text_lines):
                ax.text(x_offset, line_height+line_height*line_idx, text_line)
            
            # plot image observations
            for idx, observation in enumerate(observations['img']):
                ax = axs[int((ncols+idx)/ncols), (ncols+idx)%ncols]
                ax.clear()
                ax.axis('off')
                ax.set_title(f'Depth @ t-{idx}')
                ax.imshow(observation, cmap='grey')
            return axs
            
        plt.close(fig) # otherwise will show last frame as static plot
        animation = FuncAnimation(fig, update, frames=range(N_frames), interval=200)
        return animation
    

# an Episodic environment enhanced for reinforcment learning (has additional attributes needed for training)
class ReinforcementLearning(Episodic):
    # rewarders are additional components to Episodic parent class
        # rewarder.step() returns a reward/penalty float value
    def __init__(self, grid, spawner, actor, observer, terminators, rewarders, starting_difficulty, 
                 others=[], ckpt_freq=None, ckpt_dir=None):
        super().__init__(grid, spawner, actor, observer, terminators, 
                         others, ckpt_freq, ckpt_dir)
        self.rewarders = rewarders
        self.set_difficulty(starting_difficulty)
        self.n_episodes = 0
        self.continue_learning = True

    def stop_learning(self):
        self.continue_learning = False

    # checkpoint handling rewarders
    def checkpoint_in(self, ckpt_dir):
        super().checkpoint_in(ckpt_dir)
        for rewarder in self.rewarders:
            rewarder.checkpoint_in(ckpt_dir)
    def checkpoint_out(self):
        super().checkpoint_out()
        for rewarder in self.rewarders:
            rewarder.checkpoint_out(self.ckpt_dir)

    # we need to start the rewarders
    def start(self):
        observations, self.state = super().start()
        
        # start DRL components
        for rewarder in self.rewarders:
            rewarder.start(self)
            
        return observations, self.state
        
    # we need to step through the rewarders, and return calculated reward
    def step(self, action_value):
        observations, terminate, self.state = super().step(action_value)

        # reward function
        reward = 0
        reward_dict = {}
        for rewarder in self.rewarders:
            value = rewarder.step(self)
            reward += value
            reward_dict[rewarder.__class__.__name__] = value

        # update state with reward
        self.state['reward'] = reward
        self.state['reward_dict'] = reward_dict
        
        return observations, reward, terminate, self.state
        
    # we need to end the rewarders and evaluators
    def end(self):
        # stop DRL components
        for rewarder in self.rewarders:
            rewarder.end(self)
        # super end after reward end to trigger rewarder.end() before potential checkpoint
        super().end()