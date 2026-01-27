import warnings
warnings.filterwarnings(action='ignore', module='.*paramiko.*')
from map_tool_box.modules import Utils
import multiprocessing as mp
from pathlib import Path
import numpy as np
import paramiko
import socket
import math
import time
import os

repository_directory = Utils.get_global('repository_directory')
local_dir = Path('/home/tim/local')
dropbox_dir = Path('/home/tim/Dropbox')
progress_dir = Path(local_dir, 'progress')
overseer_dir = Path(repository_directory, 'overseer')
manager_dir = Path(overseer_dir, 'managers')
scripts_dir = Path(repository_directory, 'scripts')

# access local data
def my_instance_name():
    path = Path(local_dir, 'local_parameters.json') 
    local_params = Utils.json_read(path)
    instance_name = local_params['instance_name']
    return instance_name

# PYTHON
def run_python(py_path):
    tmux_name = py_path.split('/')[-1].split('.')[0]
    return [f'cd {repository_directory}', 'bash', f'python3 {py_path}'], tmux_name
def airsim_kill(): 
    path = Path(scripts_dir, 'kill_airsim.py') 
    return [f'python3 {path}'], 'kill'
def check_resources(instance_name=None, socket_number=8081): 
    if instance_name is None:
        instance_name = my_instance_name()
    host_ip = server_ip[instance_name]
    path = Path(scripts_dir, 'res.py') 
    return [f'python3 {path} {host_ip} {socket_number}'], 'res'
def clean_airsim(): 
    path = Path(scripts_dir, 'clean_airsim.py') 
    return [f'python3 {path}'], 'janitor'
def clean_files(): 
    path = Path(scripts_dir, 'clean_files.py') 
    return [f'python3 {path}'], 'janitor'

# DROPBOX
def dropbox_start(): 
    return ['dropbox start'], 'dropbox'
def dropbox_status(): 
    return ['dropbox status'], None
def dropbox_stop(): 
    return ['dropbox stop'], None
def dropbox_cache(): 
    path = Path(dropbox_dir, '.dropbox.cache') 
    return [f'rm -r {path}'], None
def dropbox_space(): 
    return [f'du -sh {dropbox_dir}'], None
def dropbox_pycache(): 
    return ['export PYTHONDONTWRITEBYTECODE=1'], 'dropbox'
def dropbox_exclude(file_path): 
    return [f'dropbox exclude add \'{file_path}\''], 'dropbox'
def dropbox_unexclude(file_path): 
    return [f'dropbox exclude remove \'{file_path}\''], 'dropbox'
def dropbox_listexclude(): 
    return ['dropbox exclude list'], None

# TMUX
def tmux_list(): 
    return ['tmux list-sessions'], None
def tmux_kill_all(): 
    return ['tmux kill-server'], None
def tmux_kill_session(session_name): 
    return [f'tmux kill-session -t {session_name}'], None

# GENERAL
def check_progress(): 
    return [f'ls {progress_dir}'], None
def reset_progress(): 
    return [f'rm -r {progress_dir}', f'mkdir {progress_dir}'], None
def remove_progress(key): 
    return [f'rm -r {progress_dir}{key}*'], None
def reset_temp(): 
    return [f'rm -r {local_dir}temp/', f'mkdir {local_dir}temp/'], None
def clear_trash(): 
    return [f'rm -rf {local_dir}share/Trash/*'], None
def reboot(): 
    return ['sudo reboot', 'timz'], 'reboot'
def zip(_from, _to): 
    return [f'zip -r {_to}.zip {_from}'], 'zip'
def unzip(_from, _to): 
    return [f'unzip {_from}.zip -d {_to}'], 'unzip'
def remove(_path): 
    return [f'rm -r {_path}'], 'remove'
def top():
    return ['top -b -n1'], None
def nvsmi():
    return ['nvidia-smi'], None
def disk_space():
    return ['df -H'], None
def mkdir(path):
    return [f'mkdir {path}'], None

# main functions
def tmux_get_sessions(server_name):
    return_val = execute2(server_name, tmux_list)
    if 'output' in return_val:
        return [line.split(':')[0] for line in return_val['output'].split('\n') if ':' in line]
    print('err output not in', return_val)
    return []
def tmux_kill_keyword(server_names, keyword): 
    for server_name in server_names:
        session_names = tmux_get_sessions(server_name)
        print(session_names)
        for session_name in session_names:
            if keyword in session_name:
                execute2(server_name, tmux_kill_session, job_params={'session_name':session_name})
                print('killed session', session_name, 'on server', server_name)


# server details
servers = {
    'heph':{
        'hostname' : '128.195.55.213',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/nvme0n1p2',
    },
    'magma':{
        'hostname' : '192.168.0.187',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/mapper/ubuntu--vg-ubuntu--lv',
    },
    'ace':{
        'hostname' : '128.195.54.126',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/nvme1n1p2',
    },
    'pyro':{
        'hostname' : '128.195.54.87',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/sda2',
    },
    'phoenix':{
        'hostname' : '128.195.54.85',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/sda2',
    },
    'torch':{
        'hostname' : '128.195.54.86',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/sda2',
    },
    'fox':{
        'hostname' : '128.195.55.225',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/nvme0n1p2',
    },
    'apollo':{
        'hostname' : '128.195.55.167',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/nvme0n1p2',
    },
    'flareon':{
        'hostname' : '128.195.55.161',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/nvme0n1p2',
    },
    'ninetails':{
        'hostname' : '128.195.55.208', 
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/nvme0n1p2',
    },
    'ifrit':{
        'hostname' : '128.195.55.204',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': '/dev/sda2',
    },
    'tron':{
        'hostname' : '10.8.21.241',
        'username' : 'tim',
        'password' : 'timz',
        'maindrive': 'null',
    },
    'odin':{
        'hostname' : '10.128.5.97',
        'username' : 'tjohnsen',
        'password' : 'timztimz',
        'maindrive': '/dev/nvme0n1p3',
    },
}

ip_server = {}
for server_name in servers:
    ip = servers[server_name]['hostname']
    ip_server[ip] = server_name
    
def flip_dict(other):
    flipped = {}
    for key in other:
        flipped[other[key]] = key
    return flipped
server_ip = flip_dict(ip_server)

harddrives = {server_name:servers[server_name]['maindrive'] for server_name in servers}

# SFTP
def sftp_get_or_put(server_name, get, host_path, client_path):
    server = servers[server_name]
    hostname, username, password = server['hostname'], server['username'], server['password']
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password, allow_agent=False)
    sftp = ssh.open_sftp()
    if get:
        sftp.get(client_path, host_path)
    else:
        sftp.put(host_path, client_path)

# SOCKET communications
LISTENER_SOCKET = None
def set_listener_socket(instance_name=None, socket_number=8081):
    if instance_name is None:
        instance_name = my_instance_name()
    host_ip = server_ip[instance_name]
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host_ip, socket_number))
    server_socket.listen()
    server_socket.settimeout(1)
    global LISTENER_SOCKET
    LISTENER_SOCKET = server_socket
    return server_socket

def check_listener_socket(server_socket=None):
    if server_socket is None:
        server_socket = LISTENER_SOCKET
    while True:
        try:
            client_socket, client_address = server_socket.accept()
        except:
            break

        message = client_socket.recv(1024).decode()
        ip_address = client_address[0]
        if ip_address in ip_server:
            server_name = ip_server[ip_address]
            output = {
                'server_name':server_name,
                'output':message,
                'job_func_name':'socket',
            }
            print_output(output)
            client_socket.send("Message received!".encode())
        else:
            print('missing IP', ip_address)
        client_socket.close()

# ssh execution code
def execute2(server_name, job_func, job_params={}):
    commands, tmux_name = job_func(**job_params)
    print_out = True if tmux_name is None else False
    job_func_name = str(job_func).split(' ')[1]
    return_output = {'job_func_name':str(job_func_name)}
    output = execute(server_name, commands, print_out, tmux_name)
    if output is not None:
        return_output.update(output)
    return return_output

def execute(server_name, commands, read_out=True, tmux_name=None):
    output = {}
    server = servers[server_name]
    hostname, username, password = server['hostname'], server['username'], server['password']
    output['server_name'] = server_name

    try:
        # Create SSH client
        ssh = paramiko.SSHClient()
    
        # Automatically add the server's host key (not recommended for production)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
        # Connect to the server
        ssh.connect(hostname, username=username, password=password, allow_agent=False)
    
        def send_command(command, read_out=True):
            # Execute a command on the server
            stdin, stdout, stderr = ssh.exec_command(command)
            #stdin, stdout, stderr = ssh.exec_command(f'/bin/bash -lc \"{command}\"', get_pty=True)
           
            # Print the output of the command
            if read_out:
                read = stdout.read().decode()
                return read
        
        if tmux_name is not None:
            send_command(f'tmux new-session -d -s {tmux_name}', read_out=False)
            for command in commands:
                time.sleep(1)
                tmux_command = f'tmux send-keys -t {tmux_name} \'{command}\' Enter'
                output[command] = send_command(tmux_command, read_out=read_out)
        else:
            output['output'] = send_command(';'.join(commands), read_out=read_out)
        
        # Close the SSH connection
        ssh.close()
    except Exception as e:
        print('Exception', e, 'occured from server', server_name, 'when executing commands', commands)
        output['error'] = e
    
    return output

def print_output(output):
    if len(output) > 0:
        server_name = output['server_name']
        lines = output['output']
        job_func_name = output['job_func_name']
        print(f'{server_name}  {job_func_name}:')
        if job_func_name in ['disk_space']:
            disk_name = harddrives[server_name]
            for line in lines.split('\n'):
                if disk_name in line:
                    parts = line.split(' ')
                    parts = [part for part in parts if part != '']
                    total = parts[1]
                    used = parts[2]
                    avail = parts[3]
                    print('available', avail, 'used', used, 'of', total)
        else:
            for line in lines.split('\n')[:16]:
                print(line)
        print()

def run_jobs(jobs):
    pool = mp.Pool(processes=len(jobs))
    outputs = pool.starmap(execute2, jobs)
    for output in outputs:
        if 'output' in output:
            print_output(output)
            
def run_job(working_directory, command_line_arguments, python_file_name, job_name, conda_environment=None):
    commands = []
    commands.append(f'cd {working_directory}')
    commands.append('bash')
    if conda_environment is not None:
        commands.append(f'conda activate {conda_environment}')
    command_line_arguments['job_name'] = job_name
    commands.append(f'python3 {python_file_name} {Utils.args_to_str(command_line_arguments)}')
    tmux_name = job_name
    return commands, tmux_name

class Manager:
    def expanding_map(self, base_map, device_map):
        if base_map is None:
            base_map = {}
            for server_name in device_map:
                base_map[server_name] = {}
                for device in device_map[server_name]:
                    base_map[server_name][device] = [None]*device_map[server_name][device]
        else:
            base_server_names = list(base_map.keys())
            for server_name in base_server_names:
                if server_name not in device_map:
                    del base_map[server_name]
                else:
                    for device in device_map[server_name]:
                        diff = device_map[server_name][device] - len(base_map[server_name][device])
                        if diff > 0:
                            base_map[server_name][device] = base_map[server_name][device] + [None]*diff
            for server_name in [server_name for server_name in device_map if server_name not in base_map]:
                base_map[server_name] = {}
                for device in device_map[server_name]:
                    base_map[server_name][device] = [None]*device_map[server_name][device]
        return base_map
    
    def __init__(self, manager_name, all_jobs, device_map,
                 active_jobs=None, completed_jobs=None, job_map=None, job_info=None, failed_jobs=None):
        # set base values for manager
        self.all_jobs = all_jobs
        self.device_map = device_map
        self.manager_name = manager_name
        
        # update monitoring variables
        self.active_jobs = {}
        if active_jobs is not None:
            self.active_jobs = active_jobs
        self.completed_jobs = {}
        if completed_jobs is not None:
            self.completed_jobs = completed_jobs
        self.job_info = {}
        if job_info is not None:
            self.job_info = job_info
        self.failed_jobs = {}
        if failed_jobs is not None:
            self.failed_jobs = failed_jobs
        self.n_jobs = len(self.completed_jobs) + len(self.active_jobs) + len(self.failed_jobs)
        
        # make todo jobs
        self.todo_jobs = {}
        for job in all_jobs:
            job_num = self.n_jobs + 1
            self.n_jobs += 1
            job_name = f'{manager_name}_{job_num}'
            self.todo_jobs[job_name] = job
        
        # do we need to expand maps to new devices?   
        self.job_map = self.expanding_map(job_map, device_map)
        
        # continue monitoring jobs that have been computing during offtime
        self.n_devices = 0
        for server_name in self.job_map:
            for device in self.job_map[server_name]:
                self.n_devices += len(self.job_map[server_name][device])
                for device_idx in range(len(self.job_map[server_name][device])):
                    job_name = self.job_map[server_name][device][device_idx]
                    if job_name is not None and job_name != 'BLOCKED':
                        print('continuing', job_name, 'on', server_name, 'at', Utils.get_timestamp())
                        self.job_info[job_name]['continue_time'] = Utils.get_timestamp()

        # progress tracking
        progresses = os.listdir(progress_dir)
        for progress in progresses:
            if 'manager_'+self.manager_name in progress:
                os.remove(progress_dir+progress)
        self.old_progress_path = ''
        print(len(self.todo_jobs), 'number of jobs to complete...')
        #self.save()
    
    # gets next job in queue, and returns if no more jobs exist
    def next_job(self, server_name):
        self.more_jobs = True
        if len(self.todo_jobs) <= 0:
            self.more_jobs = False
            return None, None
        next_job_name, next_job = None, None
        for job_name in self.todo_jobs:
            job = self.todo_jobs[job_name]
            if 'exlude_servers' in job:
                if server_name in job['exlude_servers']:
                    continue
            next_job_name, next_job = job_name, job
            break
        return next_job_name, next_job
    
    def start_job(self, job_name, job, server_name, device, device_idx, start_message=''):
        if 'exlude_servers' in job:
            del job['exlude_servers']
        job['job_name'] = job_name
        job['command_line_arguments']['job_name'] = job_name
        job['command_line_arguments']['device'] = device
        execute2(server_name, run_job, job)
        del self.todo_jobs[job_name]
        self.active_jobs[job_name] = job
        self.job_map[server_name][device][device_idx] = job_name
        start_time = Utils.get_timestamp()
        self.job_info[job_name] = {}
        self.job_info[job_name]['job'] = job
        self.job_info[job_name]['start_message'] = start_message
        self.job_info[job_name]['start_time'] = start_time
        self.job_info[job_name]['server_name'] = server_name
        self.job_info[job_name]['device'] = device
        self.job_info[job_name]['device_idx'] = device_idx
        print('started', job_name, 'on', server_name, 'on', device, 'at', start_time)
        
    def end_job(self, job_name, server_name, device, device_idx, end_message=''):
        execute2(server_name, tmux_kill_session, {'session_name':job_name})
        execute2(server_name, remove_progress, {'key':job_name})
        #execute2(server_name, airsim_kill)
        start_time = self.job_info[job_name]['start_time']
        end_time = Utils.get_timestamp()
        self.job_info[job_name]['end_time'] = end_time
        self.job_info[job_name]['end_message'] = end_message
        self.job_map[server_name][device][device_idx] = None
        if 'complete' in end_message:
            self.completed_jobs[job_name] = self.active_jobs[job_name]
        if 'failed' in end_message:
            self.failed_jobs[job_name] = self.active_jobs[job_name]
        del self.active_jobs[job_name]
        print('ended', job_name, 'on', server_name, 'at', end_time, 'because', end_message)

    def get_eta(self):
        delta_times = []
        for job_name in self.completed_jobs:
            completed_info = self.job_info[job_name]
            if 'end_message' not in completed_info:
                continue
            if 'continue_time' in completed_info or 'lost' in completed_info['end_message'] or 'already completed' in completed_info['end_message']:
                continue
            start_time = completed_info['start_time']
            end_time = completed_info['end_time']
            delta_time = Utils.to_datetime(end_time)-Utils.to_datetime(start_time)
            if delta_time > 0:
                delta_times.append(delta_time)
        eta_hours = -1
        n_jobs = len(self.active_jobs) + len(self.todo_jobs)
        n_rounds = math.ceil(n_jobs/self.n_devices)
        if len(delta_times) > 0:
            job_time = np.mean(delta_times)
            eta_hours = round(n_rounds*job_time/3600,2)
        return eta_hours, n_rounds
        
    def check_progress(self, server_name, device, device_idx, job_name):
        return_value = execute2(server_name, check_progress)
        if 'output' in return_value:
            output = return_value['output']
            lines = output.split('\n')
            for line in lines:
                if job_name in line:
                    line = line.replace(job_name, '')
                    if 'complete' in line:
                        self.end_job(job_name, server_name, device, device_idx, line)
                        self.update_progress()
                        return 'complete'
                    if 'failed' in line:
                        self.end_job(job_name, server_name, device, device_idx, line)
                        self.update_progress()
                        return 'failed'
        return 'null'
    
    def check_lost(self, server_name, device, device_idx, job_name):
        return_value = execute2(server_name, tmux_list)
        if 'output' in return_value:
            output = return_value['output']
            lines = output.split('\n')
            for line in lines:
                if job_name in line:
                    return False
            job = self.active_jobs[job_name]
            self.end_job(job_name, server_name, device, device_idx, 'lost')
            self.todo_jobs[job_name] = job
            return True
    
    def check_free(self, server_name, device, device_idx):
        job_name = self.job_map[server_name][device][device_idx]
        if job_name == 'BLOCKED':
            return False
        if job_name is not None:
            progress = self.check_progress(server_name, device, device_idx, job_name)
            if progress == 'complete':
                return True
            if progress == 'failed':
                return True
            lost = self.check_lost(server_name, device, device_idx, job_name)
            if lost:
                return True
            return False
        return True
    
    @staticmethod
    def load(manager_name, all_jobs=None, n_runs=None, device_map=None, 
             todo_jobs=None, active_jobs=None, completed_jobs=None, failed_jobs=None, 
             next_job_idx=None, job_map=None, job_info=None):
        manager_path = f'{manager_dir}{manager_name}/'
        print('reading manager from', manager_path)
        if all_jobs is None:
            all_jobs = Utils.pickle_read(manager_path+'all_jobs.p')
        if device_map is None:
            device_map = Utils.json_read(manager_path+'device_map.json')
        if active_jobs is None:
            active_jobs = Utils.json_read(manager_path+'active_jobs.json')
        if completed_jobs is None:
            completed_jobs = Utils.json_read(manager_path+'completed_jobs.json')
        if failed_jobs is None:
            failed_jobs = Utils.json_read(manager_path+'failed_jobs.json')
        if job_map is None:
            job_map = Utils.json_read(manager_path+'job_map.json')
        if job_info is None:
            job_info = Utils.json_read(manager_path+'job_info.json')
        return Manager(manager_name, all_jobs, device_map,
                 active_jobs, completed_jobs, job_map, job_info)
        
    def save(self, ):
        manager_path = f'{manager_dir}{self.manager_name}/'
        os.makedirs(manager_path, exist_ok=True)
        Utils.pickle_write(self.all_jobs, manager_path+'all_jobs.p')
        Utils.json_write(self.device_map, manager_path+'device_map.json')
        Utils.json_write(self.active_jobs, manager_path+'active_jobs.json')
        Utils.json_write(self.completed_jobs, manager_path+'completed_jobs.json')
        Utils.json_write(self.failed_jobs, manager_path+'failed_jobs.json')
        Utils.json_write(self.job_map, manager_path+'job_map.json')
        Utils.json_write(self.job_info, manager_path+'job_info.json')

    def update_progress(self):
        n_active = len(self.active_jobs)
        n_todo = len(self.todo_jobs)
        n_complete = len(self.completed_jobs)
        n_failed = len(self.failed_jobs)
        n_total = n_active + n_todo + n_complete + n_failed
        percent_done = int(100 * n_complete / n_total)
        eta_hours, n_rounds = self.get_eta()
        progress = f'manager_{self.manager_name} {percent_done}% {n_complete} completed {n_failed} failed {n_active} active {n_todo} todo {n_rounds} rounds  {eta_hours} hours'
        new_progress_path = f'{progress_dir}{progress}'
        if os.path.exists(self.old_progress_path):
            os.remove(self.old_progress_path)
        Utils.pickle_write(' ', new_progress_path)
        self.old_progress_path = new_progress_path
        #self.save()
        print(progress)
    
    def run(self, delay):
        time.sleep(4)
        self.update_progress()
        self.more_jobs = True
        keep_running = True
        while(keep_running):
            if len(self.active_jobs) <= 0 and len(self.todo_jobs) <= 0:
                keep_running = False
            for server_name in self.job_map:
                for device in self.job_map[server_name]:
                    for device_idx in range(len(self.job_map[server_name][device])):
                        if self.check_free(server_name, device, device_idx):
                            # do not queue up any more jobs to a server removed from the device map
                            if server_name not in self.device_map:
                                self.job_map[server_name][device][device_idx] = 'BLOCKED'
                                self.n_devices -= 1
                                #self.save()
                                break
                            # do not queue up any more jobs to a device if the number of devices has been reduced
                            if device_idx >= self.device_map[server_name][device]:
                                self.job_map[server_name][device][de_vice_idx] = 'BLOCKED'
                                self.n_devices -= 1
                                #self.save()
                                break
                            job_name, job = self.next_job(server_name)
                            if job_name is not None:
                                self.start_job(job_name, job, server_name, device, device_idx)
                            #self.save()
            time.sleep(delay)
        #self.save()
