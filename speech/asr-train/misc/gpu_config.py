import time                 
import tensorflow as tf                          # deep learning framework
import torch                                     # deep learning framework

'''
GPU Configurations

This code checks for GPU availability and prints their details. If a GPU is available, it compares the execution time of a matrix multiplication operation on both the CPU and GPU.

1. **Check GPU Availability**:
    - `check_gpu_availability()` function:
        - Lists available GPUs.
        - Prints the number of GPUs available.
        - If GPUs are available, it prints GPU details and sets a memory limit for the first GPU.
        - Catches and prints runtime errors if memory limit settings are applied after GPU initialization.
    - Example Output (TensorFlow):
        ```
        TensorFlow - GPU is Available: True
        Number of GPUs Available: 1
        TensorFlow - GPU details:  {'device_name': 'METAL'}
        1 Physical GPUs, 1 Logical GPUs
        ```

2. **Compare CPU and GPU Performance**:
    - `check_gpu_availability()` function:
        - Defines a simple matrix multiplication operation.
        - Measures and prints the execution time of the operation on the CPU.
        - If GPUs are available, it measures and prints the execution time of the operation on the GPU.
        - If no GPUs are found, it prints a corresponding message.
    - Example Output (PyTorch):
        ```
        PyTorch - CPU execution time: 0.01027 seconds
        PyTorch - MPS execution time: 0.00308 seconds
        ```

3. **Execution**:
    - Calls `check_gpu_availability('tensorflow')` and `check_gpu_availability('pytorch')` to run the above functions and display results.


'''

def check_gpu_availability(framework: str, gpu_memory_limit: float = 0.70) -> None:
    '''
    Checks for the availability of GPUs and prints their details.
    Sets memory limit for the first GPU if available.

    Args:
        framework (str): The deep learning framework to check ('tensorflow' or 'pytorch').
        gpu_memory_limit (float): The memory limit to set for the first GPU (default is 0.70).

    Returns:
        None.
    '''
    if framework.lower() == 'tensorflow':
        gpus = tf.config.list_physical_devices('GPU')
        
        print('TensorFlow - GPU is Available:', len(gpus) > 0)
        print('Number of GPUs Available:', len(gpus))

        if gpus:
            details = tf.config.experimental.get_device_details(gpus[0])
            print('TensorFlow - GPU details: ', details)

            try:
                tf.config.set_logical_device_configuration(
                    device=gpus[0],
                    logical_devices=[
                        tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit * 0.5)
                    ]
                )
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
            except RuntimeError as e:
                print('TensorFlow - Error setting GPU memory limit:', e)

    elif framework.lower() == 'pytorch':
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        if cuda_available:
            
            print('PyTorch - CUDA GPU is Available:', cuda_available)
            print('Number of CUDA GPUs Available:', torch.cuda.device_count())

            gpu_index = 0
            device = torch.device(f'cuda:{gpu_index}')
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_properties = torch.cuda.get_device_properties(device)
            print('PyTorch - CUDA GPU details: ', gpu_name, gpu_properties)

            try:
                torch.cuda.set_per_process_memory_fraction(fraction=gpu_memory_limit, device=device)
                print(f'Set PyTorch CUDA GPU memory limit to {gpu_memory_limit * 100}%')
            except Exception as e:
                print('PyTorch - Error setting CUDA GPU memory limit:', e)
        elif mps_available:
            print('PyTorch - GPU is Available:', mps_available)
            print('Number of GPUs Available:', 1)
            print('PyTorch - GPU details: MPS (Metal Performance Shaders) on Apple Silicon')
        else:
            print('PyTorch - No GPU found')

    else:
        print('Unsupported framework specified. Use "tensorflow" or "pytorch".')

def compare_cpu_gpu(framework: str) -> None:
    '''
    Compares the execution time of a matrix multiplication operation on CPU and GPU.

    Args:
        framework (str): The deep learning framework to use ('tensorflow' or 'pytorch').

    Returns:
        None.
    '''
    matrix_size = 10000

    if framework.lower() == 'tensorflow':
        a = tf.random.normal((matrix_size, matrix_size))
        b = tf.random.normal((matrix_size, matrix_size))

        # Execute on CPU
        with tf.device('/CPU:0'):
            start_time = time.time()
            c = tf.matmul(a, b)
            tf.reduce_sum(c).numpy()  # Trigger execution
            cpu_time = round(time.time() - start_time, 5)
            print(f'TensorFlow - CPU execution time: {cpu_time} seconds')

        # Execute on GPU (if available)
        if tf.config.list_physical_devices('GPU'):
            # Execute on GPU with float16 precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

            with tf.device('/GPU:0'):
                start_time = time.time()
                c = tf.matmul(a, b)
                tf.reduce_sum(c).numpy()  # Trigger execution
                gpu_time = round(time.time() - start_time, 5)
                print(f'TensorFlow - GPU execution time: {gpu_time} seconds\n')
        else:
            print('TensorFlow - No GPU found')

    elif framework.lower() == 'pytorch':
        a = torch.randn(matrix_size, matrix_size)
        b = torch.randn(matrix_size, matrix_size)

        # Execute on CPU
        start_time = time.time()
        c = torch.matmul(a, b)
        cpu_time = round(time.time() - start_time, 5)
        print(f'PyTorch - CPU execution time: {cpu_time} seconds')

        # Execute on GPU (CUDA or MPS if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        if device.type != 'cpu':
            a = a.to(device).half() # .half() casts to float16
            b = b.to(device).half() # .half() casts to float16
            
            start_time = time.time()
            c = torch.matmul(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Ensure all operations are completed
            gpu_time = round(time.time() - start_time, 5)
            print(f'PyTorch - {device.type.upper()} execution time: {gpu_time} seconds\n')
        else:
            print('PyTorch - No GPU found for comparison\n')
    else:
        print('Unsupported framework specified. Use "tensorflow" or "pytorch".\n')

# Run GPU check + compare CPU & GPU for TensorFlow and PyTorch
check_gpu_availability('tensorflow')
compare_cpu_gpu('tensorflow')

check_gpu_availability('pytorch')
compare_cpu_gpu('pytorch')
