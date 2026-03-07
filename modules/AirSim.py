from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Utils
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import subprocess
import platform
import airsim
import psutil
import math
import time
import os

def flip_euclidean_airsim(x, y, z):
    return y, x, -1*z
    
# class used to handle all things AirSim
class AirSim:

    # release path to precompiled binary to launch airsim
    # settings path to json file with requried settings
    # optional console flags
    def __init__(self, release_path, settings_name='lightweight', flags=[], timeout=10, 
                 render_animals=False, render_foilage=True, weather_type=-1, weather_degree=1,
                 additional_settings={}):
        self.release_path = release_path
        self.set_settings(settings_name, additional_settings)
        self.flags = flags
        self.timeout = timeout
        self.render_animals = render_animals
        self.render_foilage = render_foilage
        self.weather_type = weather_type
        self.weather_degree = weather_degree
        self.n_images = 0
        self.connect()
        

    # ******** MAP HANDELING ********
    def set_settings(self, settings_name, additional_settings):
        repository_dir = Utils.get_global('repository_directory')
        base_settings_path = Path(repository_dir, 'airsim', settings_name+'.json')
        settings = Utils.json_read(base_settings_path)
        settings.update(additional_settings)
        self.settings_path = Path(repository_dir, 'airsim', 'temp.json')
        Utils.json_write(self.settings_path, settings)
        
    # launch airsim map from given OS
    def connect(self, from_crash=False):
        
        # set flags
        flags = ''
        if self.flags is not None:
            flags = ' '.join(self.flags)
            
        # launch AirSim release from OS
        os_name = platform.system()
        prefix = '' if os_name == 'Windows' else 'sh '
        terminal_command = f'{prefix}{self.release_path} {flags} -settings=\"{self.settings_path}\"'
        print('issuing command:', terminal_command)
        if os_name == 'Windows':
            process = subprocess.Popen(terminal_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        elif os_name == 'Linux':
            process = subprocess.Popen(terminal_command, shell=True, start_new_session=True)
        self.pid = process.pid
                
        # wait for map to load
        time.sleep(10)
        #wait_to_continue = input()
        
        # establish communication link with airsim client
        self.client = airsim.MultirotorClient(
            timeout_value=5*60, # if no communication in this time is made then will throw TimeoutError
        )
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(1)
        self.client.takeoffAsync().join()

        # little fun critters that Microsoft added who run around and can get in the way of things
        if not self.render_animals:
            self.remove_all_animals()

        # the background trees/bushes in the horizon is labeled as Foilage and can confuse segmentation
        if not self.remove_all_foliage:
            self.remove_all_animals()
        
        # set weather
        if self.weather_type > -1:
            self.set_weather(self.weather_type, self.weather_degree)
            # add wet roads with rain
            if self.weather_type == 0:
                self.set_weather(1, self.weather_degree)
            # add snowy roads with snow
            if self.weather_type == 2:
                self.set_weather(3, self.weather_degree)
            # add leafty roads with leafs
            if self.weather_type == 4:
                self.set_weather(5, self.weather_degree)

        # wait to render
        time.sleep(2)
        #wait_to_continue = input()

        # All systems go, take off!
        #self.take_off()
        
    # clean up loaded airsim resources
    def disconnect(self):
        # this should keep child in tact to kill same process created (can handle multi in parallel)
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                pass

    def remove_all_animals(self):
        objs = self.client.simListSceneObjects()
        animals = [name for name in objs if 'Deer' in name or 'Raccoon' in name or 'Animal' in name]
        _ = [self.client.simDestroyObject(name) for name in animals] # PETA has joined the chat

    def remove_all_foliage(self):
        objs = self.client.simListSceneObjects()
        foliages = [name for name in objs if 'Foliage' in name]
        _ = [self.client.simDestroyObject(name) for name in foliages] # USFS has joined the chat

    def clear_weather(self):
        for i in range(8):
            self.client.simSetWeatherParameter(i, 0)

    def set_weather(self, weather_type, weather_degree):
        #print('setting weather', weather_type, weather_degree)
        self.client.simEnableWeather(True)
        self.client.simSetWeatherParameter(weather_type, weather_degree)

        
    # ******** DRONE KINEMATICS ********


    # the below lines are a stop_gap to fix AirSim's y-drift problem
    # see this GitHub ticket, with youtube video showing problem:
    # https://github.com/microsoft/AirSim/issues/4780
    def stabelize(self):
        self.client.rotateByYawRateAsync(0, 0.001).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.001).join()

    # check if has collided
    def check_collision(self):
        collision_info = self._airsim._client.simGetCollisionInfo()
        has_collided = collision_info.has_collided
        return has_collided 

    def get_position(self):

        # querty airsim for esimated position
        pos = self.client.getMultirotorState().kinematics_estimated.position
        
        # change dumb drone coords y, x, -z to euclidean coords x, y, z 
        x, y, z = flip_euclidean_airsim(pos.x_val, pos.y_val, pos.z_val)
        return Data_Structure.Point(x, y, z)

    # get rotation about the z-axis (yaw), returns in radians between -pi to +pi
    def get_yaw(self):

        # querty airsim for esimated quaternions
        q = self._airsim._client.getMultirotorState().kinematics_estimated.orientation
        
        # convert quaternions to eularian angles
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        
        return yaw
        
    def move(self, x_rel, y_rel, z_rel, speed=2, stabelize=True):

        # change euclidean coords x, y, z to dumb drone coords y, x, -z
        x_rel, y_rel, z_rel = flip_euclidean_airsim(x_rel, y_rel, z_rel)

        # get current position then move relative
        pos = self.client.getMultirotorState().kinematics_estimated.position
        current_position = np.array([pos.x_val, pos.y_val, pos.z_val])
        target_position = current_position + np.array([x_rel, y_rel, z_rel])
        #print('cur', current_position, 'rel', np.array([x_rel, y_rel, z_rel]), 'tar', target_position)

        self.client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 
                                        speed, timeout_sec = self.timeout).join()
        #self._airsim._client.moveByVelocityAsync(x_rel, y_rel, z_rel, speed).join()
        
        # stabalize drone?
        if stabelize:
            self.stabelize()


    # teleports to position
        # x,y,z in euclidean (will convert to drone coords)
        # yaw, pitch, roll in radians
    def teleport(self, x, y, z, yaw, pitch=0, roll=0, ignore_collision=True, stabelize=True):

        # change euclidean coords x, y, z to dumb drone coords y, x, -z
        x, y, z = flip_euclidean_airsim(x, y, z)

        # create new airsim pose object
        pose = airsim.Pose(
            airsim.Vector3r(x, y, z), 
            airsim.to_quaternion(pitch, roll, yaw)
        )

        # directly set to new pose object
        self.client.simSetVehiclePose(pose, ignore_collision=ignore_collision)
        
        # stabalize drone?
        if stabelize:
            self.stabelize()
            
    def take_off(self):
        self.client.takeoffAsync(timeout_sec = self.timeout).join()

        
    # ******** SENSOR MECHANICS ********

    # camera_view values:
        # 'front_center' or '0'
        # 'front_right' or '1'
        # 'front_left' or '2'
        # 'bottom_center' or '3'
        # 'back_center' or '4'
    # image_type values:
        # Scene = 0, 
        # DepthPlanar = 1, 
        # DepthPerspective = 2, >>> use this for depth maps
        # DepthVis = 3, 
        # DisparityNormalized = 4,
        # Segmentation = 5,
        # SurfaceNormals = 6,
        # Infrared = 7,
        # OpticalFlow = 8,
        # OpticalFlowVis = 9
    def camera(self, camera_view='0', image_type=2, compress=False, view_img=False, out_dir=None,
              make_channel_first=True):
        if image_type in [1, 2, 3, 4]:
            as_float = True
            is_gray = True
            is_image = False
        else:
            as_float = False
            is_gray = False
            is_image = True
        image_request = airsim.ImageRequest(camera_view, image_type, as_float, compress)
        img_array = []
        while len(img_array) <= 0: # loop for dead images (happens some times)
            response = self.client.simGetImages([image_request])[0]
            if as_float:
                np_flat = np.array(response.image_data_float, dtype=float)
            else:
                np_flat = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            if is_gray:
                img_array = np.reshape(np_flat, (response.height, response.width))
            else:
                img_array = np.reshape(np_flat, (response.height, response.width, 3))
                    
        if view_img:
            if is_gray:
                plt.imshow(img_array, cmap='grey', vmin=0, vmax=255)
            else:
                plt.imshow(img_array)
            plt.show()
            
        self.n_images += 1
        if out_dir is not None:
            out_path = Path(out_dir, f'img_{self.n_images}.jpg')
            img_array[img_array<0] = 0
            img_array[img_array>255] = 255
            img_array = img_array.astype(np.uint8)
            image = Image.fromarray(img_array)
            image.save(out_path)

        # make channel-first
        if make_channel_first and len(img_array) > 0:
            if is_gray:
                img_array = np.expand_dims(img_array, axis=0)
            else:
                img_array = np.moveaxis(img_array, 2, 0)
                    
        return img_array


    # ******** OTHER FUNCTIONS ********

    def write_voxels(self, center, super_cube_res, sub_cube_res, output_path):
        center = airsim.Vector3r(*center)
        self.client.simCreateVoxelGrid(center, super_cube_res, super_cube_res, super_cube_res, sub_cube_res, output_path)

            