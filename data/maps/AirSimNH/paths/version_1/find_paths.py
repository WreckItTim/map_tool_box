from map_tool_box.modules import Data_Map
from map_tool_box.modules import Action
from map_tool_box.modules import Astar
from map_tool_box.modules import Utils
from pathlib import Path
import random
import sys
import os

map_name = 'AirSimNH'
motion_type = 'dpad'
random_seed = 7
magnitudes = [2, 4, 8, 16, 32]
n_paths = 1_000
ckpt_frequency = 10
max_iterations = 100_000
job_name = 'null'

# required arg -- subdir:value (str) to write found paths to (see below)
if len(sys.argv) > 1:
    arguments = Utils.parse_arguments(sys.argv[1:])
    locals().update(arguments)

# set file IO paths from arg subdir
DATA_DIR = Utils.get_global('data_directory')
subdir_path = Path(DATA_DIR, 'maps', map_name, 'paths', subdir)
os.makedirs(subdir_path, exist_ok=True)
file_name = f'path_list__{random_seed}.p'
file_path = Path(subdir_path, file_name)

# read paths from file?
if os.path.exists(file_path):
    path_list = Utils.pickle_read(file_path)
else:
    path_list = []

# set global random seed -- used to sample start and target points
Utils.set_random_seed(random_seed)

# link map to find shortest path in
data_map = Data_Map.DataMapRoof(map_name)

# specify types of actions that we can take
actions = []

# can move forward, backward, left, right
if motion_type == 'dpad':
    for magnitude in magnitudes:
        actions.append(Action.Move(Action.step_forward, magnitude))
        actions.append(Action.Move(Action.step_backward, magnitude))
        actions.append(Action.Move(Action.step_left, magnitude))
        actions.append(Action.Move(Action.step_right, magnitude))
    heuristic = Astar.dpad_heursitic
    heuristic_kwargs = {
        'max_x':max(magnitudes),
        'max_y':max(magnitudes), 
        'max_z':1,
    }

# can only make forward-facing motions, thus has to rotate first before changing direction of motion
if motion_type == 'forward':
    for magnitude in magnitudes:
        actions.append(Action.Forward(magnitude))
    actions.append(Action.RotateLeft())
    actions.append(Action.RotateRight())
    heuristic = Astar.forward_heursitic
    heuristic_kwargs = {
        'max_xy':max(magnitudes),
        'max_z':1,
    }

# default costs of 1 per each action
costs = [1]*len(actions)

# make Astar object used to find paths
pather = Astar.Astar(data_map, actions, costs, heuristic, heuristic_kwargs)

# find path
while len(path_list) < n_paths:
    start_position, target_position = data_map.sample_points(sample_size=2)
    start_direction = random.randint(0, 3)
    path, iterations, result = pather.search(start_position, start_direction, target_position,
                                             max_iterations=max_iterations)
    print('astar search result =', result, 'in', iterations, 'iterations')
    if 'goal' in result:
        writable_path = [[node.x, node.y, node.z, node.direction, node.action] for node in path]
        writable_path = [iterations, writable_path]
        path_list.append(writable_path)
        if len(path_list)%ckpt_frequency == 0 or len(path_list) == n_paths:
            Utils.pickle_write(file_path, path_list)
            progress = f'found {len(path_list)} paths of {n_paths}'
            Utils.update_progress(job_name, progress)
Utils.update_progress(job_name, 'complete')