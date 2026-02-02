from jtop import jtop
from time import time

# input arguments:
    # n_calls is the number of times to call the function
    # function is a custom python func that defines what it is you want to benchmark
    # func_params are kwargs to pass into above function
# return values:
    # jetson_stats as retruned from jtop (the key-values here will change depending on the NVIDIA device used)
    # latency which is the stop_time-start_time to function call
def run_benchmark(n_calls, func, func_params):

    # initialize benchmarking
    jetson = jtop() # create benchmark logging thread
    jetson.start() # start benchmark window
    start_time = time()

    # make function calls
    for run in range(n_calls):  
        func(**func_params)

    # clean up benchmarking
    latency = time() - start_time
    jetson.close() # close benchmark logging thread
    jetson_stats = jetson.stats.copy()

    return jetson_stats, latency