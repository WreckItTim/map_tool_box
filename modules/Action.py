from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Component

# dummy template
class Action(Component.Component):
    # grid is used to determine if valid point, typically using something like:    
        # valid = (not grid.in_object(x, y, z)
        #      and grid.in_bounds(x, y, z))
    def act(self, grid, point):
        raise NotImplementedError

    # step from env, update x,y,z,dir
    def step(self, environment):
        grid = environment.get_grid()
        point = environment.get_point()
        point, valid = self.act(grid, point)
        environment.set_point(point)
        return valid

# these helper functions only consider stepping through translational movements (x,y,z)
def step_through(grid, x, y, z, step_function, magnitude, step_size, grid_xy_resolution):
    x3, y3, z3 = x, y, z
    distance_traveled = 0
    valid = True
    while(distance_traveled < magnitude):
        x2, y2, z2 = step_function(x, y, z, step_size)
        valid = (grid.in_bounds(x2, y2, z2)
             and not grid.in_object(x2, y2, z2))
        if not valid:
            break
        x, y, z = x2, y2, z2
        if x%grid_xy_resolution==0 and y%grid_xy_resolution==0:
            x3, y3, z3 = x, y, z
        distance_traveled += step_size
    return x3, y3, z3, valid
def MoveForward(x, y, z, step_size=1):
    return x, y+step_size, z
def MoveBackward(x, y, z, step_size=1):
    return x, y-step_size, z
def MoveLeft(x, y, z, step_size=1):
    return x-step_size, y, z
def MoveRight(x, y, z, step_size=1):
    return x+step_size, y, z
def MoveUp(x, y, z, step_size=1):
    return x, y, z+step_size
def MoveDown(x, y, z, step_size=1):
    return x, y, z-step_size
def step_forward(x, y, z, step_size=1):
    return x, y+step_size, z
def step_backward(x, y, z, step_size=1):
    return x, y-step_size, z
def step_left(x, y, z, step_size=1):
    return x-step_size, y, z
def step_right(x, y, z, step_size=1):
    return x+step_size, y, z
def step_up(x, y, z, step_size=1):
    return x, y, z+step_size
def step_down(x, y, z, step_size=1):
    return x, y, z-step_size
    
class Move(Action):
    def __init__(self, move_function, magnitude, step_size=1, grid_xy_resolution=1):
        self.move_function = move_function
        self.magnitude = magnitude
        self.step_size = step_size
        self.grid_xy_resolution = grid_xy_resolution

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        x, y, z, valid = step_through(grid, x, y, z, self.move_function, 
                                     self.magnitude, self.step_size, self.grid_xy_resolution)

        return Data_Structure.Point(x, y, z, direction), valid
    def __str__(self):
        return f'{self.move_function.__name__}{self.magnitude}'
    def __repr__(self):
        return str(self)

class Forward(Action):
    def __init__(self, magnitude, step_size=1, grid_xy_resolution=1):
        self.magnitude = magnitude
        self.step_size = step_size
        self.grid_xy_resolution = grid_xy_resolution

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        # check which way to move
        if direction == 0: # move forward
            step_function = MoveForward
        elif direction == 1: # move right
            step_function = MoveRight
        elif direction == 2: # move backward
            step_function = MoveBackward
        elif direction == 3: # move left
            step_function = MoveLeft
            
        x, y, z, valid = step_through(grid, x, y, z, step_function, 
                                     self.magnitude, self.step_size, self.grid_xy_resolution)

        return Data_Structure.Point(x, y, z, direction), valid
    def __str__(self):
        return f'Forward{self.magnitude}'
    def __repr__(self):
        return str(self)

class StrafeRight(Action):
    def __init__(self, magnitude, step_size=1, grid_xy_resolution=1):
        self.magnitude = magnitude
        self.step_size = step_size
        self.grid_xy_resolution = grid_xy_resolution

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        # check which way to move
        if direction == 0: # move right
            step_function = MoveRight
        elif direction == 1: # move backward
            step_function = MoveBackward
        elif direction == 2: # move left
            step_function = MoveLeft
        elif direction == 3: # move forward
            step_function = MoveForward
            
        x, y, z, valid = step_through(grid, x, y, z, step_function, 
                                     self.magnitude, self.step_size, self.grid_xy_resolution)

        return Data_Structure.Point(x, y, z, direction), valid
    def __str__(self):
        return f'StrafeRight{self.magnitude}'
    def __repr__(self):
        return str(self)

class StrafeLeft(Action):
    def __init__(self, magnitude, step_size=1, grid_xy_resolution=1):
        self.magnitude = magnitude
        self.step_size = step_size
        self.grid_xy_resolution = grid_xy_resolution

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        # check which way to move
        if direction == 0: # move left
            step_function = MoveLeft
        elif direction == 1: # move forward
            step_function = MoveForward
        elif direction == 2: # move right
            step_function = MoveRight
        elif direction == 3: # move backward
            step_function = MoveBackward
            
        x, y, z, valid = step_through(grid, x, y, z, step_function, 
                                     self.magnitude, self.step_size, self.grid_xy_resolution)

        return Data_Structure.Point(x, y, z, direction), valid
    def __str__(self):
        return f'StrafeLeft{self.magnitude}'
    def __repr__(self):
        return str(self)
    
class RotateClockwise(Action):

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        direction += 1
        if direction == 4:
            direction = 0

        return Data_Structure.Point(x, y, z, direction), True
    def __str__(self):
        return f'RotateClockwise'
    def __repr__(self):
        return str(self)
    
class RotateCounter(Action):

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        direction -= 1
        if direction == -1:
            direction = 3

        return Data_Structure.Point(x, y, z, direction), True
    def __str__(self):
        return f'RotateCounter'
    def __repr__(self):
        return str(self)
    
class RotateLeft(Action):

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        direction += 1
        if direction == 4:
            direction = 0

        return Data_Structure.Point(x, y, z, direction), True
    def __str__(self):
        return f'RotateClockwise'
    def __repr__(self):
        return str(self)
    
class RotateRight(Action):

    def act(self, grid, point):
        x, y, z, direction = point.unpack()
        
        direction -= 1
        if direction == -1:
            direction = 3

        return Data_Structure.Point(x, y, z, direction), True
    def __str__(self):
        return f'RotateCounter'
    def __repr__(self):
        return str(self)