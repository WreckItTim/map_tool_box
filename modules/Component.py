# base template to use for an environmental component
    # functions should be defined by child if there is some use case for them
    # mostly used in DRL or control applications
    # each method inputs state dictionary, and optionaly updates it
    # state dictionary is generic and up to the implementation to define key,value pairs in it
class Component:
    def start(self, environment):
        pass
    def step(self, environment):
        pass
    def end(self, environment):
        pass