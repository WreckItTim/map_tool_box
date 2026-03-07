from map_tool_box.modules import Component

# translates action-value to actions from RLenvironment
class Actor(Component.Component):
    def __init__(self, actions):
        self.actions = actions

    def start(self, environment):
        for action in self.actions:
            action.start(environment)

    def end(self, environment):
        for action in self.actions:
            action.end(environment)

# chooses one action from those available
class Discrete(Actor):
    
    def step(self, environment):
        action_value = environment.get_action_value()
        action = self.actions[int(action_value)]
        action.step(environment)