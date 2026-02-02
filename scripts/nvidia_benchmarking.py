from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Environment
from map_tool_box.modules import Data_Map
from map_tool_box.modules import Control
from map_tool_box.modules import Spawner
from map_tool_box.modules import Astar
from map_tool_box.modules import Utils
from map_tool_box.modules import Model
from IPython.display import HTML
from pathlib import Path

# point to model you wish to evaluate
models_directory = Utils.get_global('models_directory') # check to make sure this is correct on your local computer (it should be auto)
model_name = 'AirSim_Navigation DRL_beta' # change to proper model name if needed (name is also sub-directory path)
model_subdir = model_name.replace(' ', '/')
model_directory = Path(models_directory, model_subdir)

# read control params and make objects from config.py file
config_path = Path(model_directory, 'config.py')
with open(config_path) as f:
    src = f.read()
    code = compile(src, "__main__.py", "exec")
    exec(code)

# read model from file
model_path = Path(model_directory, 'model.zip')
model = Model.read_model(model_path)

# set parameters for which set of Astar paths to evaluate on
set_name = 'test' # train val test
n_paths = 100 # if None then will read all paths from file, otherwise an integer value specifying number of paths PER DIFFICULTY
difficulties = None # [i for i in range(2, 17)] # if None then will read all difficulties from file, otherwise expects a list of difficulty keys

# read paths from file (usese some variables read in from config.py file) 
    # -- you can overwrite map_name or astar_version, but the default values or those used to train the model
paths = Astar.read_curriculum(map_name, astar_version, set_name, n_paths, difficulties)

# create spawner object to iterate through paths from environment
spawner = Spawner.CurricululmEval(paths)

# create environment that we will step through (uses some objects read in from config.py file)
environment = Environment.Episodic(data_map, spawner, actor, observer, terminators)

# run entire evaluation over set astar paths and model
# turned flag off for collecting observations since it is memory heavy
episodes_states, accuracy = Control.eval(environment, model, save_additional_state_info=True, save_observations=True)

# accuracy is auto calculated for you, however you can iterate through the states to recalculate the same accuracy for when state['end']=='Goal'
print(f'accuracy = {accuracy*100:.2f}')
