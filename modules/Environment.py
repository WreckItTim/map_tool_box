from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
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
    # spawner, actions, observer, and terminators are components
        # spawner.start() sets initial environment x, y, z, direction, (target) values 
        # actor.step() updates intermediate environment x, y, z, direction values
        # observer.step() returns a processed observation captured at current state
        # terminate.step() returns a boolean determining if we should end the episode
    # grid object uses geo x,y,z,direction coordinates to determine collisions and out-of-bounds
        # if grid is a Data_Map object then this is used to access cached observations at given coordinates as well
    # save_states is a flag for efficiency (True will take slighly longer and use more memory, see implemenation)
    def __init__(self, grid, spawner, actor, observer, terminators, save_states=False):
        self.grid = grid
        self.spawner = spawner
        self.actor = actor
        self.observer = observer
        self.terminators = terminators
        self.save_states = save_states
        self.n_episodes = 0

    
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

    # returns expected number of steps for current path
    # this func assumes path is set (by spawner)
    def get_path(self):
        return self.path   
    def set_path(self, path):
        self.path = path
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
        self.actor.start(self)
        self.observer.start(self)
        for terminator in self.terminators:
            terminator.start(self)
            
        # observation space
        observations = copy.deepcopy(self.observer.step(self))

        # initial state variables
        state = {}
        if self.save_states:
            state['step'] = 0
            state['action_value'] = 'start'
            state['point'] = self.point
            state['target'] = self.target
            state['path_steps'] = self.get_path_steps()
            
        return observations, state
        
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
            
        # observation space
        observations = copy.deepcopy(self.observer.step(self))

        # check if we should terminate episode
        terminate = False
        for terminator in self.terminators:
            terminate = terminator.step(self)
            if terminate:
                terminator_reason = terminator.__class__.__name__
                break

        # track state variables
        state = {}
        if self.save_states:
            state['step'] = self.n_steps
            state['action_value'] = action_value
            state['point'] = self.point # set by actor
        if terminate:
            state['end'] = terminator_reason
        
        return observations, terminate, state

    # end of epsiode
    def end(self):
        self.n_episodes += 1
        
        # end all components
        self.spawner.end(self)
        self.actor.end(self)
        self.observer.end(self)
        for terminator in self.terminators:
            terminator.end(self)

    # this is a slightly hard-coded method used for debugging
    # returns an animation of an episode based on given list of states
    def animate_episode(self, states, show_rewards=True):

        # get some params
        last_point = states[0]['point']
        target_point = states[0]['target']
        path_steps = states[0]['path_steps']
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
            target_point = self.get_target()
            ax.scatter(target_point.x+self.grid.x_shift, target_point.y+self.grid.y_shift, color='green', 
                       marker='*', s=64)
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
            text_lines.append(f'goal: {target_point}')
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
    # rewarders and evaluators are additional components to Episodic parent class
        # rewarder.step() returns a reward/penalty float value
        # evaluator.start()/step()/end() are arbitrary DRL-specific classes to call and do user-defined tasks
    def __init__(self, grid, spawner, actor, observer, rewarders, terminators, 
                 evaluators=[], starting_difficulty=None, save_states=False):
        super().__init__(grid, spawner, actor, observer, terminators, save_states)
        self.rewarders = rewarders
        self.evaluators = evaluators
        self.set_difficulty(starting_difficulty)
        self.start_learning()

    def start_learning(self):
        self.n_episodes = 0
        self.continue_learning = True
    def stop_learning(self):
        self.continue_learning = False

    # we need to start the rewarders and evaluators
    def start(self):
        observations, state = super().start()
        
        # start DRL components
        for rewarder in self.rewarders:
            rewarder.start(self)
        for evaluator in self.evaluators:
            evaluator.start(self)
            
        return observations, state
        
    # we need to step through the rewarders and evaluators, and return calculated reward
    def step(self, action_value):
        observations, terminate, state = super().step(action_value)

        # reward function
        reward = 0
        reward_dict = {}
        for rewarder in self.rewarders:
            value = rewarder.step(self)
            reward += value
            reward_dict[rewarder.__class__.__name__] = value

        # update state with reward
        if self.save_states:
            state['reward'] = reward
            state['reward_dict'] = reward_dict

        # any evaluations on step?
        for evaluator in self.evaluators:
            evaluator.step(self)
        
        return observations, reward, terminate, state
        
    # we need to end the rewarders and evaluators
    def end(self):
        super().end()
        self.n_episodes += 1
        
        # stop DRL components
        for rewarder in self.rewarders:
            rewarder.end(self)
        for evaluator in self.evaluators:
            evaluator.end(self)