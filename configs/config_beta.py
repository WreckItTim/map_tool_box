import numpy as np

map_name = 'AirSimNH'
action_magnitudes = [1, 2, 4, 8, 16, 32]
n_history = 3 # attention mechanism that keeps track of this many previous observations
goal_tolerance = 4 # maximum distance (meters) required to reach goal within
steps_multiplier = 4 # multiplies by a path's optimal number of steps to determine maximum number of steps
horizon = 255 # farthest perceivable distance (meters) -- used for normalization
roof_name = 'default'
depth_sensor_name = 'DepthV1'
astar_version = 'experimental'
grid_xy_resolution = 1 # collected discrete data points at this resolution

# enviornment map -- grid data map to collect observations and detect collisions / out of bounds
from map_tool_box.modules import Data_Map
data_map = Data_Map.DataMapRoof(map_name, roof_name=roof_name)

# action space -- only move forward in current direction, can rotate yaw to change direction
from map_tool_box.modules import Action
from map_tool_box.modules import Actor
actions = []
actions.append(Action.RotateClockwise())
actions.append(Action.RotateCounter())
# # forward motion
# for magnitude in action_magnitudes:
#     actions.append(Action.Forward(magnitude))
# # dpad motion
for magnitude in action_magnitudes:
    actions.append(Action.StrafeRight(magnitude, grid_xy_resolution=grid_xy_resolution))
for magnitude in action_magnitudes:
    actions.append(Action.StrafeLeft(magnitude, grid_xy_resolution=grid_xy_resolution))
for magnitude in action_magnitudes:
    actions.append(Action.Forward(magnitude, grid_xy_resolution=grid_xy_resolution))
actor = Actor.Discrete(actions)

# observation space -- n-many forward facing depth maps
from map_tool_box.modules import Data_Transformation
from map_tool_box.modules import Observer
from map_tool_box.modules import Sensor
img_observer = Observer.Observer([
        #Sensor.DataMapSensor(data_map, depth_sensor_name, Data_Transformation.Normalize(max_input=horizon)), # with normalization
        Sensor.DataMapSensor(data_map, depth_sensor_name), # no normalization
    ], n_history=n_history, data_type=np.uint8)
vec_observer = Observer.Observer([
        #Sensor.GoalDisplacement(Data_Transformation.Normalize(min_input=-1*horizon, max_input=horizon)), # with normalization
        Sensor.RelativeGoal(), # self normalizes since r, theta are different scales
        Sensor.DistanceBounds(data_map, Data_Transformation.Normalize(max_input=horizon)), # with normalization
        Sensor.CurrentDirection(Data_Transformation.Normalize(max_input=3)), # with normalization
        #Sensor.DistanceBounds(), # no normalization
        #Sensor.GoalDisplacement(), # no normalization
        #Sensor.CurrentDirection(), # no normalization
    ], n_history=n_history)
observer_dict = {
    'vec':vec_observer,
    'img':img_observer,
}
observer = Observer.DictObserver(observer_dict)


# termination criteria (episodic) -- how does an episode end?
from map_tool_box.modules import Terminator
terminators = [
    Terminator.Goal(goal_tolerance),
    Terminator.MaxSteps(steps_multiplier),
]

