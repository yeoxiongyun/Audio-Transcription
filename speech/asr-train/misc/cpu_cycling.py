import os
import psutil
import time
import multiprocessing
import numpy as np
import torch
import tensorflow as tf

def print_thread_info():
    '''
    Prints system threading information, including:
    - Max CPU threads
    - OpenMP thread settings
    - NumPy, PyTorch, and TensorFlow threading configurations
    '''
    print('Max CPU threads:', multiprocessing.cpu_count())
    print('OMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS', 'Not set'))
    
    print('NumPy Max Threads:', np.__config__.show())
    
    print('PyTorch Max Threads:', torch.get_num_threads())
    
    print('TensorFlow Max Threads (Inter-Op):', tf.config.threading.get_inter_op_parallelism_threads())
    print('TensorFlow Max Threads (Intra-Op):', tf.config.threading.get_intra_op_parallelism_threads())

def cycle_cpu_affinity(pid: int, interval: int = 600):
    '''
    Cycles the process across different CPU cores every `interval` seconds.

    Parameters:
    - pid (int): Process ID of the target process.
    - interval (int): Time in seconds between CPU core swaps.
    '''
    cpu_cores = list(range(psutil.cpu_count()))  # Get available CPU cores
    print(f'Managing CPU affinity for PID: {pid} across cores: {cpu_cores}')

    while True:
        for core in cpu_cores:
            try:
                p = psutil.Process(pid)
                p.cpu_affinity([core])  # Bind process to a single core
                print(f'🔄 Moved process {pid} to CPU core {core}')
                time.sleep(interval)  # Wait before switching to the next core
            except psutil.NoSuchProcess:
                print(f'❌ Process {pid} no longer exists.')
                return
            except Exception as e:
                print(f'⚠️ Error: {e}')

# Limit OpenMP threading to 1 thread
os.environ['OMP_NUM_THREADS'] = '1'

# Print thread information
print_thread_info()

### Example: Swapping cores every 10 minutes for the current process
##current_pid = os.getpid()
##cycle_cpu_affinity(current_pid, interval=600)  # 600 seconds = 10 minutes
