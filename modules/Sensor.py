from map_tool_box.modules import Component
import numpy as np

# parent class -- must define a sense() function that returns a numpy array
    # should also define a get_null() function that defines how to return a missing observation
class Sensor(Component.Component):
    
    def sense(self):
        raise NotImplementedError
        
    def get_null(self):
        raise NotImplementedError

# gets data from map object (observations are key-value pairs saved in memory)
class DataMapSensor(Sensor):
    
    def __init__(self, data_map, sensor_name, data_transformation=None):
        self.data_map = data_map
        self.sensor_name = sensor_name
        self.data_transformation = data_transformation

    def get_shape(self):
        sensor_meta = self.data_map.get_sensor_meta(self.sensor_name)
        sensor_shape = sensor_meta['shape']
        return sensor_shape

    def step(self, environment):
        point = environment.get_point()
        observation = self.sense(point)
        return observation

    def sense(self, point):
        observation = self.data_map.get_data_point(point, self.sensor_name)
        if self.data_transformation is not None:
            observation = self.data_transformation.transform(observation)
        return observation

# gets x,y,z displacement from goal
class GoalDisplacement(Sensor):
    
    def __init__(self, data_transformation=None):
        self.data_transformation = data_transformation

    def get_shape(self):
        return (3,)
    
    def step(self, environment):
        goal_displacement = environment.get_goal_displacement()
        observation = self.sense(goal_displacement)
        return observation

    def sense(self, goal_displacement):
        observation = goal_displacement
        if self.data_transformation is not None:
            observation = self.data_transformation.transform(observation)
        return np.array(observation)

# gets current direction of agent
class CurrentDirection(Sensor):
    
    def __init__(self, data_transformation=None):
        self.data_transformation = data_transformation

    def get_shape(self):
        return (1,)

    def step(self, environment):
        point = environment.get_point()
        direction = point.direction
        observation = self.sense(direction)
        return observation

    def sense(self, direction):
        observation = direction
        if self.data_transformation is not None:
            observation = self.data_transformation.transform(observation)
        return np.array([observation])

# gets data from map object (observations are key-value pairs saved in memory)
class RelativeGoal(Sensor):
    
    def __init__(self, self_normalize=True, data_transformation=None):
        self.self_normalize = self_normalize
        self.data_transformation = data_transformation

    def get_shape(self):
        return (2,)
    
    def step(self, environment):
        goal_displacement = environment.get_goal_displacement()
        observation = self.sense(goal_displacement)
        return observation

    def sense(self, goal_displacement):

        # get polar coordinates from displacement
        r = np.linalg.norm(goal_displacement)
        theta = np.arctan2(goal_displacement[1], goal_displacement[0])

        # self normalize since r and theta are on different scales
        if self.self_normalize:
            r = np.interp(r, (0, 255), (0.1, 1))
            theta = np.interp(theta, (-np.pi, np.pi), (0.1, 1))
        
        observation = np.array([r, theta], dtype=np.float32)
        if self.data_transformation is not None:
            observation = self.data_transformation.transform(observation)
        return np.array(observation)

# gets data from map object (observations are key-value pairs saved in memory)
class DistanceBounds(Sensor):
    
    def __init__(self, grid, data_transformation=None):
        self.grid = grid
        self.data_transformation = data_transformation

    def get_shape(self):
        return (1,)
    
    def step(self, environment):
        point = environment.get_point()
        observation = self.sense(point)
        return observation

    def sense(self, point):
        x, y, z, direction = point.unpack()
        if direction == 0:
            xy_distance = abs(y - (self.grid.y_max-1))
        elif direction == 1:
            xy_distance = abs(x - (self.grid.x_max-1))
        elif direction == 2:
            xy_distance = abs(y - self.grid.y_min)
        else:# direction == 3:
            xy_distance = abs(x - self.grid.x_min)
        
        observation = np.array([xy_distance], dtype=np.float32)
        if self.data_transformation is not None:
            observation = self.data_transformation.transform(observation)
        return np.array(observation)