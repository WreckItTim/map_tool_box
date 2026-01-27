from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Data_Map
from map_tool_box.modules import Action
from map_tool_box.modules import Utils
from pathlib import Path
import numpy as np
import math
import os

DATA_DIR = Utils.get_global('data_directory')

#### ASTAR SHORTEST PATH

def get_quick_path(start_point, target_point, map_name='AirSimNH', start_direction=0, max_iterations=10_000):
    data_map = Data_Map.DataMapRoof(map_name, memory_saver=True)
    actions = []
    magnitudes = [2, 4, 8, 16, 32]
    for magnitude in magnitudes:
        actions.append(Action.Forward(magnitude))
    actions.append(Action.RotateLeft())
    actions.append(Action.RotateRight())
        
    costs = [1]*len(actions)
    heuristic = forward_heursitic
    heuristic_kwargs = {
        'max_xy':max(magnitudes),
        'max_z':1,
    }
    goal_tolerance = 0
    astar = Astar(data_map, actions, costs, heuristic, heuristic_kwargs, goal_tolerance=goal_tolerance)
    path, outer_iterations, result = astar.search(start_point, target_point, max_iterations=max_iterations)
    valid = result in ['goal']
    if result in ['goal']:
        print('path succesfully found')
    if result in ['iterations']:
        print('max iterations reached, no valid path may exist or is too complex. Try passing in the parameter into get_quick_path max_iterations=value where value is > 10_000')
    if result in ['children']:
        print('no valid path')

    return path, valid

# reads in dictionaries of curriculum paths seperated by train, val, test and keyed by level
# optionally processes it to get maximum number of paths from each level
def read_curriculum(map_name, astar_version, set_name, n_paths=None, difficulties=None):
    curriculum_dir = Path(DATA_DIR, 'maps', map_name, 'paths', astar_version, 'curriculum')
    # read astar from old repo format
    if astar_version in ['old', 'experimental']:
        paths_dict = {}
        for difficulty in difficulties:
            paths_dict[difficulty] = []
            fname = f'level_{difficulty}_split_{set_name}.p'
            fpath = Path(curriculum_dir, fname)

            # this is a numpy array 
            paths_in = Utils.pickle_read(fpath)

            # convert to Point objects
            for path in paths_in:
                target_pos = path[0, :3]
                points = []
                for i in range(1, len(path)):
                    pose = path[i, :4]
                    point = Data_Structure.Point(*pose)
                    points.append(point)
                points[-1] = Data_Structure.Point(*target_pos)
                paths_dict[difficulty].append(points)
    else:
        paths_dict = Utils.pickle_read(Path(curriculum_dir, f'{set_name}_dict.p'))
        if n_paths is not None:
            for level in paths_dict:
                if difficulties is not None and level not in difficulties:
                    continue
                paths_dict[level] = paths_dict[level][:n_paths]
    return paths_dict

# reads in all path_list parts from given map and version
def read_paths(map_name, astar_version):
    astar_dir = Path(DATA_DIR, 'maps', map_name, 'paths', astar_version)
    paths = []
    for file_name in os.listdir(astar_dir):
        if 'path_list' not in file_name:
            continue
        fpath = Path(astar_dir, file_name)
        path_list = Utils.pickle_read(fpath)
        for idx, path_info in enumerate(path_list):
            iterations = path_info[0]
            path = path_info[1]
            paths.append(path)
    return paths

# calculates the componentized (x, y, z) euclidean distances to goal
def component_distance_to_goal(current, goal, goal_tolerance=0):
    distances = np.maximum(np.abs(goal-current)-goal_tolerance, 0)
    return distances

# caluclates the number of moves required to reach given distance
# assuming each move travels at max magnitude units 
def number_moves(distance, magnitude):
    return 0 if distance == 0 else math.ceil(distance/magnitude)
    
# considers actions move in a d-pad able to move the maximum distance
# max_x, max_y, max_z are kwargs and are the maximum distance the actions can make
def dpad_heursitic(point, goal, goal_tolerance, max_x, max_y, max_z):
    x, y, z, direction = point.unpack()
    # get component distance to goal
    x_distance, y_distance, z_distance = component_distance_to_goal(np.array([x, y, z]), np.array(goal), goal_tolerance)

    # get minimum number of translational movements required to reach goal
    n_x = number_moves(x_distance, max_x)
    n_y = number_moves(y_distance, max_y)
    n_z = number_moves(z_distance, max_z)
    h = n_x + n_y + n_z
    
    return h

# considers can only make forward movements, and must rotate yaw to change direction
# max_xy and max_z are kwargs and are the maximum distance the forward motion can make
# assumes 90 degree yaw rotations
def forward_heursitic(point, goal, goal_tolerance, max_xy, max_z):
    x, y, z, direction = point.unpack()
    # get component distance to goal
    x_distance, y_distance, z_distance = component_distance_to_goal(np.array([x, y, z]), np.array(goal), goal_tolerance)

    # get minimum number of translational movements required to reach goal
    n_x = number_moves(x_distance, max_xy)
    n_y = number_moves(y_distance, max_xy)
    n_z = number_moves(z_distance, max_z)
    h = n_x + n_y + n_z

    # get number of rotational movements required to reach goal
    if x_distance > 0:
        # not facing right but goal is on right
        if direction != 1 and x < goal[0]:
            h += 1
        # not facing left but goal is on left
        elif direction != 3 and goal[0] < x:
            h += 1
    if y_distance > 0:
        # not facing forward but goal is forward
        if direction != 0 and y < goal[1]:
            h += 1
        # not facing backward but goal is backward
        elif direction != 2 and goal[1] < y:
            h += 1
        
    return h

class AStarNode:
    def __init__(self, point, action=None, cost=1, parent=None):
        self.parent = parent
        self.x = point.x
        self.y = point.y
        self.z = point.z
        self.position = np.array([point.x,point.y,point.z])
        self.direction = point.direction
        self.action = action
        self.cost = cost
        self.g = 0
        self.h = 0
        self.f = 0
        self.name = f'{self.x}_{self.y}_{self.z}_{self.direction}'
    def unpack(self):
        return self.x, self.y, self.z, self.direction
        
class Astar:
    # grid is an object used to determine if actions are valid given current x, y, z, direction
    # actions is a list of custom action objects:
        # x, y, z, direction, collision = action.act(grid, x, y, z, direction)
    # costs is a list of numerical values that represent the cost of each action (same idx as above)
    # heuristic is a function that estimates the cost to the goal
    # goal tolerance is minimum distance from target pos required for success
    def __init__(self, grid, actions, costs, heuristic, heuristic_kwargs, goal_tolerance=0):
        self.grid = grid
        self.actions = actions
        self.costs = costs
        self.goal_tolerance = goal_tolerance
        self.heuristic = heuristic
        self.heuristic_kwargs = heuristic_kwargs.copy()
        self.n_actions = len(actions)
    
    # get path found
    def return_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            point = Data_Structure.Point(current.x, current.y, current.z, current.direction, current.action)
            path.append(point)
            current = current.parent
        # reverse order to return nodes ordered from start to end
        path = path[::-1]
        return path
        
    # search for optimal path
    # returns a key stating the termination reason
    def search(self, start_point, target_point, max_iterations=10_000):
        goal = np.array([target_point.x, target_point.y, target_point.z])
        start_node = AStarNode(start_point)
        to_visit = {}
        visited = {}
        to_visit[start_node.name] = start_node

        outer_iterations = 0
            
        while len(to_visit) > 0:
            outer_iterations += 1    

            # get best node to look at next
            best_f = np.inf
            for name in to_visit:
                node = to_visit[name]
                if node.f < best_f:
                    current_node = node
                    best_f = node.f

            # bail if too many iterations
            if outer_iterations > max_iterations:
                return self.return_path(current_node), outer_iterations, 'iterations'

            # move from to_visit to visited
            del to_visit[current_node.name]
            visited[current_node.name] = current_node

            # test if goal is reached or not, if yes then return the path
            if np.linalg.norm(current_node.position - goal) <= self.goal_tolerance:
                return self.return_path(current_node), outer_iterations, 'goal'

            # generate children from all valid actions
            children = []
            for i in range(self.n_actions):
                action = self.actions[i]
                cost = self.costs[i]
                
                point, valid = action.act(self.grid, current_node)
    
                # check valid move
                #print(i, x, y, z, direction, valid)
                if not valid:
                    continue
                    
                # check if visited already
                name = f'{point.x}_{point.y}_{point.z}_{point.direction}'
                if name in visited:
                    continue 

                # valid child, add to list
                child = AStarNode(point, action, cost, current_node)
                children.append(child)
            #print('children', len(children))
            # update to_visit list with proposed children
            for child in children:

                # calculate current cost
                child.g = current_node.g + child.cost
                
                # estimate remaining cost
                child.h = self.heuristic(child, goal, self.goal_tolerance, **self.heuristic_kwargs)

                # estimate total cost                    
                child.f = child.g + child.h

                # check against other child
                if child.name in to_visit and child.g > to_visit[child.name].g:
                    continue

                # add child to to_visit, overwriting previous child with worse g
                to_visit[child.name] = child
                
        return self.return_path(current_node), outer_iterations, 'children'
