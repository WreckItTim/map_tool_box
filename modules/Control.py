import random
import copy

SUCCESS_REASONS = ['Goal']

# step through an entire episode with environment, generationg actions from model
# environment is an Episodic class object in the Environment.py module
# model can be any object that has a predict() function as defined below
# save_states and save_observations are flags for saving additional information as needed
def play_episode(environment, model, save_additional_state_info=False, save_observations=False):
    
    # tell environment to save state variables at each step or not
    save_states_before = environment.save_states
    environment.save_states = save_additional_state_info

    # start episode
    observations, state = environment.start()

    # initialize and log state variables
    if save_observations: # include observations? (this is memory heavy)
        state['observations'] = copy.deepcopy(observations)
    states = [state]
    
    # iterate through each step
    terminate = False
    while(not terminate):
        
        # get action from model based on observations
        action_value = model.predict(observations)
    
        # take step based on action value
        returned_values = environment.step(action_value)
        if len(returned_values) == 3:
            observations, terminate, state = returned_values
        if len(returned_values) == 4:
            observations, reward, terminate, state = returned_values
        if len(returned_values) == 5:
            observations, reward, terminate, truncate, state = returned_values
        state['action_value'] = action_value

        # log state variables
        if save_observations: # include observations? (this is memory heavy)
            state['observations'] = copy.deepcopy(observations)
        states.append(state)

    # end episode
    environment.end()

    # reset environment flag back to what it was before
    environment.save_states = save_states_before
    
    return states

# step through several episodes with environment and model
# calculates navigation accuracy
def eval(environment, model, save_additional_state_info=False, save_observations=False):

    # reset spawner -- sets any vars as needed to check if there are more paths to evaluate
    environment.spawner.reset()

    # keep playing a new episode while spawner has more paths to evaluate on
    episodes_states = []
    n_success = 0
    while(environment.spawner.has_more_paths()):

        # step though single episode
        states = play_episode(environment, model, save_additional_state_info, save_observations)

        # aggregate results of episode
        episodes_states.append(states)
        terminator_reason = states[-1]['end']
        if terminator_reason in SUCCESS_REASONS:
            n_success += 1

    # calculate accuracy of all episodes
    accuracy = n_success / len(episodes_states)
    
    return episodes_states, accuracy

# model that selects a random discrete action index
class RandomDiscrete:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, observations):
        return random.randint(0, self.n_actions-1)