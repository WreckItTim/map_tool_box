from map_tool_box.modules import Data_Transformation, Utils
from map_tool_box.modules import Data_Structure
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from time import localtime, time  
from pathlib import Path 
import numpy as np
import random
import math
import os

DATA_DIR = Utils.get_global('data_directory')

# get and write meta data from data dictionaries for given map and sensor
# this leads to quicker load times during execution of data fetches
def write_file_map(map_name, sensor_name):
    sensor_dir = Path(DATA_DIR, 'maps', map_name, 'sensors', sensor_name)
    file_map = {}
    file_names = os.listdir(sensor_dir)
    for file_name in file_names:
        if 'data_dict__' not in file_name:
            continue
        file_path = Path(sensor_dir, file_name)
        data_dict = Utils.pickle_read(file_path)
        for x2 in data_dict:
            if x2 not in file_map:
                file_map[x2] = {}
            for y2 in data_dict[x2]:
                file_map[x2][y2] = file_path
    file_map_path = Path(sensor_dir, 'file_map.p')
    Utils.pickle_write(file_map_path, file_map)

# write json sensor meta data to file 
def write_sensor_meta(map_name, sensor_name, sensor_meta):
    sensor_dir = Path(DATA_DIR, 'maps', map_name, 'sensors', sensor_name)
    sensor_meta_path = Path(sensor_dir, 'meta.json')
    Utils.json_write(meta_path, sensor_meta_path)
                        
# these helper functions are used to transform direction to yaw and backwards
def fix_yaw(yaw):
    while yaw < 0:
        yaw += 2*np.pi
    while yaw >= 2*np.pi:
        yaw -= 2*np.pi
    return yaw
def yaw_to_direction(yaw):
    yaw = fix_yaw(yaw)
    direction = round(yaw/(np.pi/2))
    return direction
def direction_to_yaw(direction):
    yaw = direction * np.pi/2
    return yaw

# reads a datamap__..._.p from file
def read_data_dict(map_name, input_sensor, part_name):
    file_name = f'data_dict__{part_name}.p'
    file_path = Path(DATA_DIR, 'maps', map_name, 'sensors', input_sensor, file_name)
    data_dict = Utils.pickle_read(file_path)
    return data_dict

# writes a datamap__..._.p to file 
def write_data_dict(map_name, output_sensor, data_dict, part_name):
    sensor_dir = Path(DATA_DIR, 'maps', map_name, 'sensors', output_sensor)
    os.makedirs(sensor_dir, exist_ok=True)
    file_name = f'data_dict__{part_name}.p'
    file_path = Path(sensor_dir, file_name)
    Utils.pickle_write(file_path, data_dict)

# apply given transformation to all observations in given data_dict
def transform_data_dict(data_dict, transformation, transformation_parameters={}):
    
    # transform each observation in input data map
    output_data = {}
    for x in data_dict:
        output_data[x] = {}
        for y in data_dict[x]:
            output_data[x][y] = {}
            for z in data_dict[x][y]:
                output_data[x][y][z] = {}
                for d in data_dict[x][y][z]:
                    observation = data_dict[x][y][z][d]
                    transformed = transformation(observation, **transformation_parameters)
                    output_data[x][y][z][d] = transformed
    return output_data


# reads all data dicts from file at given Path('maps', map_name, 'sensors', input_sensor)
# transforms all data dicts and writes to file at Path('maps', map_name, 'sensors', output_sensor)
def transform_sensor(map_name, input_sensor, output_sensor, transformation, transformation_parameters={}):
    
    # make any required directories for output path
    input_dir = Path(DATA_DIR, 'maps', map_name, 'sensors', input_sensor)
    output_dir = Path(DATA_DIR, 'maps', map_name, 'sensors', output_sensor)
    os.makedirs(output_dir, exist_ok=True)
    
    # load one data_dict at a time
    for file_name in os.listdir(input_dir):
    
        # set paths and transform data
        if 'data_dict' in file_name:
            input_path = Path(input_dir, file_name)
            output_path = Path(output_dir, file_name)
            
            # read input data
            input_data = Utils.pickle_read(input_path)
            
            # transform each observation in input data map
            output_data = transform_data_dict(input_data, transformation, transformation_parameters)
            
            # write transformed data
            Utils.pickle_write(output_path, output_data)


# helper function to get axis from pyplot subplots axs regardless of nrows and ncols
    # this mitigates issue when nrows=1 and returns a 1d array of axs
def get_ax(axs, row, col, nrows, ncols):
    if nrows > 1:
        ax = axs[row, col]
    else:
        ax = axs[col]
    return ax


# base class data map
    # use RoofDataMap child (leaving room for future children)
class DataMap:

    
    # memory_saver=False will cache all data dicts as they are read, leading to quicker fetch times but more memory
    # memory_saver=True will only cache the most recent data dict, leading to less memory but longer fetch times
    def __init__(self, map_name, memory_saver=False):
        self.map_name = map_name
        self.file_map = {} # maps x,y,z,d points to file path of data dictionary
        self.memory_saver = memory_saver # only cache the most recent data dict?
        self._last_filepath = '' # keep track of file path to last read data dict for memory saver
        self.loaded_paths = {} # keep track of all data dicts loaded into file
        self.data_dicts = {} # actual data_dicts data at sensor_name, x, y, z, d

    # return dictionary of various meta data saved in local meta.json file
    def get_sensor_meta(self, sensor_name):
        sensor_dir = Path(DATA_DIR, 'maps', self.map_name, 'sensors', sensor_name)
        sensor_meta_path = Path(sensor_dir, 'sensor_meta.json')
        sensor_meta = Utils.json_read(sensor_meta_path)
        return sensor_meta
    
    # detects if grid space at given points is considered occupied
    def in_object(self, x, y, z):
        raise NotImplementedError

        
    # checks if position is inside of given bounds
    def in_bounds(self, x, y, z):
        raise NotImplementedError

    
    # plots 2d representation of map to pyplot axis
    def plot_map(self, fig, ax, show_z=True, cmap='hot', interval = 40):
        raise NotImplementedError

    
    # loads grid and sets x_shift and y_shift for translating between numpy array and grid map
    def set_grid(self):
        raise NotImplementedError

    # shows path on map
    # points is list of objects that must have given attributes: x, y, z, direction
    def view_path(self, points, fig=None, ax=None):
        make_and_show = False
        if fig is None:
            make_and_show = True
            fig, ax = plt.subplots()
        start_point = points[0]
        target_point = points[-1]
        self.plot_map(fig, ax)
        for point in points:
            self.plot_agent(fig, ax, point,
                 start_point=start_point, target_point=target_point, show_fov=False)
        if make_and_show:
            plt.show()
    
    # plots position of agent on map along with current path if not None
    # assumes plot_map() was called before this, of which set the x and y ticks
    def plot_agent(self, fig, ax, point, fov_horizion = 255,
                 start_point=None, target_point=None, path=None, marker_scale=1, show_fov=False):
        base_marker_size = 32
        x, y, z, direction = point.unpack()
        
        if direction is not None:

            # arrow markers that indicate direction
            direction_markers = {
                0:10,
                1:9,
                2:11,
                3:8,
            }
            
            # plot agent's position and indicate direction
            ax.scatter(x+self.x_shift, y+self.y_shift, color='cyan', marker=direction_markers[direction], s=base_marker_size*marker_scale)
            
            # plot agent's field of view
            if show_fov:
                if direction == 0: a, b, c, d = -1, 1, 1, 1
                if direction == 1: a, b, c, d = 1, 1, 1, -1
                if direction == 2: a, b, c, d = -1, -1, 1, -1
                if direction == 3: a, b, c, d = -1, 1, -1, -1
                fov1_x_points = [x+self.x_shift+a*i for i in range(fov_horizion)]
                fov1_y_points = [y+self.y_shift+b*i for i in range(fov_horizion)]
                fov2_x_points = [x+self.x_shift+c*i for i in range(fov_horizion)]
                fov2_y_points = [y+self.y_shift+d*i for i in range(fov_horizion)]
                ax.plot(fov1_x_points, fov1_y_points, color='cyan', linestyle='--')
                ax.plot(fov2_x_points, fov2_y_points, color='cyan', linestyle='--')
            
        else:       
            
            # plot only the agent's position
            ax.scatter(x+self.x_shift, y+self.y_shift, color='cyan', marker='.', s=base_marker_size*marker_scale)
        
        # plot path
        if start_point is not None:
            ax.scatter(start_point.x+self.x_shift, start_point.y+self.y_shift, color='blue', marker='x', s=base_marker_size*marker_scale)
        if target_point is not None:
            ax.scatter(target_point.x+self.x_shift, target_point.y+self.y_shift, color='green', marker='*', s=2*base_marker_size*marker_scale)
        if path is not None:
            for next_point in path:
                ax.scatter(next_point.x+self.x_shift, next_point.y+self.y_shift, color='blue', marker='x', s=base_marker_size*marker_scale)

    
    # return an animation object with data specified at each panel
    # sensor_names is list of keys to use from sensor_metas and observations
    # sensor_metas is json dict at each value
    # observations is dictionary of observations at each value (key by sensor name and indexed by panel index)
    # sensor_pseudonames is dictionary linking sensor_name to displayed panel title
    def animate(self, sensor_metas, observations, nrows, ncols,
                sensor_names=None, points=None, show_fov=False,
                sensor_psuedonames={}, fig_scale=4, text_blocks=None, show_path=False):
        if sensor_names is None:
            sensor_names = list(observations.keys())
        N_frames = len(observations[sensor_names[0]])
        show_map = points is not None
        
        # Create a figure and axes for all frames
        if show_map:
            layout = [['map'] + ['.' for _ in range(1, ncols)]]
            height_ratios = [3]
            for row in range(1, nrows):
                layout.append([f'{row},{col}' for col in range(ncols)])
                height_ratios.append(2)
        else:
            layout = []
            height_ratios = []
            for row in range(nrows):
                layout.append([f'{row},{col}' for col in range(ncols)])
                height_ratios.append(1)
        fig, axs = plt.subplot_mosaic(layout, 
                                      #figsize=(fig_scale*ncols,fig_scale*nrows), 
                                      #gridspec_kw={'height_ratios':height_ratios,
                                     )
        #fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(fig_scale*ncols,fig_scale*nrows))
        if show_map:
            ax = axs['map']
            self.plot_map(fig, ax, show_z=True)
            ax.set_title(f'Aerial View') 
            ax.set_aspect('equal') 
        fig.tight_layout()
        
        # Animation function
        path = []
        start_point = points[0]
        target_point = points[-1]
        def update(t):
            
            # view map
            if show_map:
                point = points[t]
                ax = axs['map']
                ax.clear()
                self.plot_map(fig, ax, show_z=False)
                if show_path:
                    self.plot_agent(fig, ax, point, show_fov=show_fov,
                         start_point=start_point, target_point=target_point, path=path)
                    path.append(point)
                else:
                    self.plot_agent(fig, ax, point, show_fov=show_fov)
                ax.set_title(f'Aerial View') # dir for d-pad direction -- 0,1,2,3
                ax.set_aspect('equal') 
                
            # view observations
            offset = 0 # shift panel number by this offset
            if show_map:
                offset = ncols # show aerial view in top row only
            for sensor_idx, sensor_name in enumerate(sensor_names):
                sensor_meta = sensor_metas[sensor_name]
                observation = observations[sensor_name][t]
                
                # overlay on previous frame?
                if 'overlay' in sensor_meta and sensor_meta['overlay']:
                    offset -= 1

                # get axis corresponding to this panel to paint on
                panel_idx = sensor_idx + offset
                col = panel_idx % ncols
                row = int(panel_idx/ncols)
                ax = axs[f'{row},{col}']
                #ax = get_ax(axs, row, col, nrows, ncols)

                # prepare panel
                if 'overlay' not in sensor_meta or not sensor_meta['overlay']:
                    ax.clear()                 
                    ax.set_visible(True)
                    ax.set_aspect('equal')

                # paint observation onto panel
                if observation is None:
                    ax.set_visible(False)
                else:
                    visualization_type = sensor_meta['visualization_type']
                    if 'segmentation' in visualization_type:
                        ax.imshow(observation_temp, cmap='rainbow', vmin=0, vmax=64)
                    elif 'rgb-image' in visualization_type:
                        ax.imshow(Data_Transformation.channel_first_to_last(observation))
                    elif 'grey-image' in visualization_type:
                        if len(observation.shape) == 3:
                            ax.imshow(observation[0], cmap='grey')
                        else:
                            ax.imshow(observation, cmap='grey')
                    elif 'masks' in visualization_type:
                        largest_val, largest_name = -1, None
                        for mask_name in observation:
                            mask = observation[mask_name]
                            val = np.sum(mask)
                            if val > largest_val:
                                largest_val = val
                                largest_name = mask_name
                        ax.imshow(observation[largest_name], vmin=0, vmax=1)
                    elif 'rectangles' in visualization_type:
                        for rectangle_name in observation:
                            x_min, y_min, width, height = observation[rectangle_name]
                            rectangle = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='white', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(x_min, y_min, rectangle_name, color='white', fontsize=8)

                # set title name of panel
                panel_name = sensor_name
                if sensor_name in sensor_psuedonames:
                    panel_name = sensor_psuedonames[sensor_name]
                ax.set_title(panel_name)
                
            # turn off dead panels
            for col2 in range(col+1, ncols):
                ax = axs[f'{row},{col}']
                #ax = get_ax(axs, row, col2, nrows, ncols)    
                ax.clear()
                ax.set_visible(False)
                    
            # add text blocks to bottom-right most panel
            if text_blocks is not None:
                sensor_idx2 = 1+sensor_idx+offset
                col = ncols-1
                row = nrows-1
                ax = axs[f'{row},{col}']
                #ax = get_ax(axs, row, col, nrows, ncols)    
                ax.set_visible(True)
                text_block = text_blocks[t]
                ax.clear()
                ax.imshow(np.full([36, 64], 255).astype(np.uint8), cmap='grey', vmin=0, vmax=255)
                line_height = 5
                for line_idx, text_line in enumerate(text_block):
                    ax.text(0, line_height+line_height*line_idx, text_line)
                ax.axis('off')
                ax.set_aspect('equal')
                ax.set_title('State')
    
            # turn off ticks
            start_row = 0
            if show_map:
                start_row = 1
            for row in range(start_row, nrows):
                for col in range(ncols):
                    ax = axs[f'{row},{col}']
                    #ax = get_ax(axs, row, col, nrows, ncols) 
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
            #return tuple(axs.flatten())
            return axs
        
        # Create the animation
        plt.close(fig) # otherwise will show last frame as static plot
        ani = FuncAnimation(fig, update, frames=range(N_frames), interval=200)
        return ani

    
    # clear up memory
        # sensor_name=None will clear all memory, otherwise only clear mem associated with sensor_name
    def clear_cache(self, sensor_name=None):
        self.fetched_data.clear()
        if sensor_name is None:
            self.data_dicts.clear()
            self.loaded_paths.clear()
        else:
            self.data_dicts[sensor_name].clear()
            del self.data_dicts[sensor_name]
            del self.loaded_paths[sensor_name]    

            
    # map file paths of data dictionaries to [sensor_name, x, y] keys
        # NOTE: this is an efficiency choice that considers data_maps are made by chunking only x and y and not z and d
    def get_filepath(self, sensor_name, x, y):
        if sensor_name not in self.file_map:
            self.file_map[sensor_name] = {}
            sensor_dir = Path(DATA_DIR, 'maps', self.map_name, 'sensors', sensor_name)
            # get meta data from file -- linking each x,y coordinate to data_dict part
            file_map_path = Path(sensor_dir, 'file_map.p')
            if not os.path.exists(file_map_path):
                write_file_map(self.map_name, sensor_name)
            file_map = Utils.pickle_read(file_map_path)
            self.file_map[sensor_name] = file_map
        exists = sensor_name in self.file_map and x in self.file_map[sensor_name] and y in self.file_map[sensor_name][x]
        if exists:
            return self.file_map[sensor_name][x][y]
        else:
            return None

        
    # reads corresponding data_dict_part into memory if single data point does not exist in current data_dict
    # if (self.memory_saver or use_memory_saver) then does not store data_dict_part into memory
    def get_data_point(self, point, sensor_name, use_memory_saver=False):
        #print('get', sensor_name, point)
        observation = None  
        x, y, z, direction = point.unpack()
        
        # use memory saver which only checks most recent loaded data dict from file
        if self.memory_saver or use_memory_saver:
            filepath = self.get_filepath(sensor_name, x, y)
            # single cache system
            if self._last_filepath != filepath:
                self._last_filepath = filepath
                self._last_data_dict = Utils.pickle_read(filepath)
            data_exists = x in self._last_data_dict and y in self._last_data_dict[x]# and z in data_dict_part[x][y] and direction in data_dict_part[x][y][z]
            if data_exists:
                observation = self._last_data_dict[x][y][z][direction]
            return observation

        # else keep track of all data_dicts from file in memory (much quicker)
        if sensor_name not in self.data_dicts:
            self.data_dicts[sensor_name] = {}
            self.loaded_paths[sensor_name] = []
        data_dict = self.data_dicts[sensor_name]
        data_exists = x in data_dict and y in data_dict[x]# and z in data_dict[x][y] and direction in data_dict[x][y][z]
        if data_exists:
            observation = data_dict[x][y][z][direction]
        else:
            filepath = self.get_filepath(sensor_name, x, y)
            if filepath not in self.loaded_paths[sensor_name]:
                data_dict_part = Utils.pickle_read(filepath)
                self.loaded_paths[sensor_name].append(filepath)
                #print(f'{sensor_dir}{file_name}')
                #print(data_dict_part.keys())
                # need to do a deep update 
                for _x in data_dict_part:
                    #print(data_dict_part[_x].keys())
                    if _x not in data_dict:
                        data_dict[_x] = {}
                    for _y in data_dict_part[_x]:
                        #print(data_dict_part[_x][_y].keys())
                        if _y not in data_dict[_x]:
                            data_dict[_x][_y] = {}
                        for _z in data_dict_part[_x][_y]:
                            #print(data_dict_part[_x][_y][_z])
                            #print(data_dict_part[_x][_y][_z].keys())
                            if _z not in data_dict[_x][_y]:
                                data_dict[_x][_y][_z] = {}
                            for _direction in data_dict_part[_x][_y][_z]:
                                #print(data_dict_part[_x][_y][_z][_direction].shape)
                                data_dict[_x][_y][_z][_direction] = data_dict_part[_x][_y][_z][_direction]
            data_exists = x in data_dict and y in data_dict[x] #and z in data_dict[x][y] and direction in data_dict[x][y][z]
            if data_exists:
                observation = data_dict[x][y][z][direction]
        return observation


    
            
    #### METHODS FOR DATA FETCHING
    

    # creates a pyplot animation from dictionary of observations at given points
        # {sensor_name:[obs_1, ..., obs_N]}
    # sensor_names is list of same length oas observations linking to data_map sensor names
    # points=[] is an optional list of [x,y,z,d] points to display position on map at each obs
    def make_animation(self, observations, sensor_names, points=None, ncols=2, sensor_psuedonames={}, 
                             fig_scale=4, text_blocks=None, show_path=False, show_fov=False):
        
        # keep track of how many panels to show in animation
        show_map = points is not None
        n_panels = show_map*ncols
        
        # get meta from each sensor that keeps track of display settings
        sensor_metas = {}
        for idx, observation_name in enumerate(observations):
            sensor_name = sensor_names[idx]
            sensor_meta = self.get_sensor_meta(sensor_name)
            sensor_metas[observation_name] = sensor_meta
            # overlay=True means it will write this panel ontop of the previous one during animation
            if 'overlay' not in sensor_meta or not sensor_meta['overlay']:
                n_panels += 1

        # calculate number of rows in animation
        nrows = math.ceil(n_panels/ncols)

        # make pyplot animation
        animation = self.animate(sensor_metas, observations, nrows, ncols, 
                                 points=points, show_fov=show_fov, sensor_psuedonames=sensor_psuedonames, 
                                 fig_scale=fig_scale, text_blocks=text_blocks, show_path=show_path)
        
        return animation
        
    
    # input points desired to grab = [ [x_1, y_1, z_1, direction_1], ..., [x_N, y_N, z_N, direction_N]] 
    # returns fetched_data which is a dictionary organized as:
        # 'points':[ [x_1, y_1, z_1, direction_1], ..., [x_N, y_N, z_N, direction_N]]
        # 'observations':
            # sensor_name: np.array([ observation_at_point_1, ..., observation_at_point_N])
    # if make_animation=True then also returns a pyplot animation of data
    def fetch_data_at_points(self, sensor_names, points):

        # 0. order points for quicker fetch times, since data is stored in data_dict chunks
            # keep track of original order
        ordered_points = {}
        for idx, point in enumerate(points):
            filepath = self.get_filepath(sensor_names[0], point.x, point.y)
            if filepath not in ordered_points:
                ordered_points[filepath] = []
            ordered_points[filepath].append((idx, point))
            
        # 1. start data fetch
        sensor_metas = {} # keeps track of json file meta data about each sensor
        observations = {} # keeps track of actual observations fetched from each sensor
        for sensor_name in sensor_names:
            sensor_meta = self.get_sensor_meta(sensor_name)
            sensor_metas[sensor_name] = sensor_meta
            observations[sensor_name] = [None]*len(points)

        # 2. collect observations at each point
        for sensor_name in sensor_names:
            for filepath in ordered_points:
                pairs = ordered_points[filepath]
                for pair in pairs:
                    idx, point = pair
                    observation = self.get_data_point(point, sensor_name)
                    observations[sensor_name][idx] = observation

        # 3. end data fetch
        for sensor_name in sensor_names:
            observations[sensor_name] = np.array(observations[sensor_name])

        return observations
    
    
    # randomly grabs data points from grid
        # each data point is valid (not in object)
        # no data point is repeated
        # each data point has data for all sensors passed in        
    def sample_data(self, sensor_names, sample_size):

        # get pool of possible points
        pool = []
        z = 4 # TODO: adjust for 3d motion, currently considers z is locked at 4
        for x in range(self.x_min, self.x_max):
            for y in range(self.y_min, self.y_max):
                # check if data exists for all sensors
                valid_point = True
                for sensor_name in sensor_names:
                    if self.get_filepath(sensor_name, x, y) is None:
                        valid_point = False
                        break
                if not valid_point:
                    continue
                for d in range(4):
                    pool.append(Data_Structure.Point(x, y, z, d))

        assert sample_size <= len(pool), f'sample size of {sample_size} is too large, only {len(pool)} # of points available'
        
        # sample total number of points
        sampled_points = random.sample(pool, sample_size)

        # fetch data
        fetched_data = self.fetch_data_at_points(sensor_names, sampled_points)

        return sampled_points, fetched_data
    

# considers an occupancy grid where each x,y position has a z value
    # the z value corresponds to the highest collidable z-position at that x,y-position
    # all positions at a z-value <= that at x,y are considered occupied and those above are unoccupied
# this leads to speed up optimizations in collision detection
# considers that the roofs are a rectangluar shape where the same y values exist for every x, and uniform spacing for both x and y
class DataMapRoof(DataMap):
    # child constructor 
    def __init__(self, map_name, collision_threshold=2, roof_name='default', memory_saver=False):
        super().__init__(map_name, memory_saver)
        roofs_path = Path(DATA_DIR, 'maps', map_name, 'roofs', roof_name+'.p')
        self.set_roofs(roofs_path)
        self.collision_threshold = collision_threshold

    # reads and sets roofs object from file
    def set_roofs(self, roofs_path):
    
        # read roofs from file
        self.roofs_path = roofs_path
        self.roofs = Utils.pickle_read(roofs_path)
        
        # set x stats
        self.xs = list(self.roofs.keys())
        self.x_min = np.min(self.xs) # inclusive
        self.x_max = np.max(self.xs) + 1 # exclusive
        self.x_n = len(self.xs)
        self.x_delta = (self.x_max - self.x_min) / (self.x_n)
        
        # set y stats
        self.ys = list(self.roofs[self.x_min].keys())
        self.y_min = np.min(self.ys) # inclusive
        self.y_max = np.max(self.ys) + 1 # exclusive
        self.y_n = len(self.ys)
        self.y_delta = (self.y_max - self.y_min) / (self.y_n)

        # make numpy array of roofs
        self.roofs_array = np.zeros((self.x_n, self.y_n))
        for xi in range(self.x_n):
            x = self.x_min + xi*self.x_delta
            for yi in range(self.y_n):
                y = self.y_min + yi*self.y_delta
                self.roofs_array[xi, yi] = self.roofs[x][y]

        # set array shifts to translate between dictionary and array of positions
        self.x_shift = -1*self.x_min
        self.y_shift = -1*self.y_min
    
    # checks if position is in object based on collision_threshold (meters) above rooftop
    def in_object(self, x, y, z):
        return z <= self.get_roof(x, y) + self.collision_threshold
        
    # checks if position is inside of given bounds
    def in_bounds(self, x, y, z):
        return (x >= self.x_min and x < self.x_max
                and y >= self.y_min and y < self.y_max)

    def get_roof(self, x, y):
        return self.roofs[x][y]
        
    # plots 2d representation of map to pyplot axis
    def plot_map(self, fig, ax, show_z=True, cmap='hot', interval = 40, shrink=1):
        # plot roofs onto map
        im = ax.imshow(self.roofs_array.T, cmap=cmap, origin='lower') # , interpolation='nearest'

        # show height of each roof as color bar
        if show_z:
            cbar = fig.colorbar(im, shrink=shrink)
            cbar.ax.get_yaxis().labelpad = 25
            cbar.ax.set_ylabel('z [meters]', rotation=270)

        # set viewing window
        ax.set_xlim(0, self.x_n)
        ax.set_ylim(0, self.y_n)

        # translate array index values to x,y positions
        ax.set_xticks([i for i in range(0, self.x_n, interval)], 
                      [int(self.x_min + i*self.x_delta) for i in range(0, self.x_n, interval)], rotation=90)
        ax.set_yticks([i for i in range(0, self.y_n, interval)], 
                      [int(self.y_min + i*self.y_delta) for i in range(0, self.y_n, interval)])
    