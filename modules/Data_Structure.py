
# direction is typically used as a paramter but technically is optional
# action is optional parameter that defines the previous Action taken to get to this point
class Point:
    def __init__(self, x, y, z, direction=None, action=None):
        self.x = x
        self.y = y
        self.z = z
        self.direction = direction
        self.action = action
    def unpack(self):
        return self.x, self.y, self.z, self.direction
    def __str__(self):
        x, y, z, direction = self.unpack()
        if direction is None:
            s = f'x:{x} y:{y} z:{z}'
        else:
            s = f'x:{x} y:{y} z:{z} d:{direction}'
        return s
    def __repr__(self):
        return str(self)