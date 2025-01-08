import multiprocessing as mp
import platform
import sys
import subprocess
import os
import pkg_resources
from typing import List

# List of specified packages to check
example_specified_packages = ['conda', 'pip', 'kivy', 'pyinstaller']

# Utility Function: Run shell commands and return output
def run_command(command: List[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT).decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        return f'Error: {e.output.decode("utf-8")}'

# Utility Function: Print Installed Packages
def print_installed_packages() -> str:
    '''Retrieve and write installed packages to a file.'''
    try:
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode('utf-8').strip()
        return installed_packages
    except subprocess.CalledProcessError:
        print('Error retrieving installed packages.')
        return ''

# Utility Function: Get Package Versions
def get_package_version(package: str) -> str:
    '''Get the version of a specified package.'''
    try:
        version = subprocess.check_output([sys.executable, '-m', 'pip', 'show', package]).decode('utf-8')
        for line in version.splitlines():
            if line.startswith('Version:'):
                return line.split()[-1]
        return 'not installed'
    except subprocess.CalledProcessError:
        return 'not installed'

#  Utility Function: Calculate Container Size
def calc_container(path: str) -> int:
    '''Calculate the size of a directory.'''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Utility Function: Get Package Information
def get_package_info(package: str) -> str:
    '''
    Get the version and size (in KB) of a package.
    Parameters:
        package: str - Name of the package.
    Returns:
        A formatted string containing package name, version, and size.
    '''
    version = get_package_version(package)
    size = 'not installed'
    try:
        distribution = pkg_resources.get_distribution(package)
        if distribution.location:
            path = os.path.join(distribution.location, package)
            size = calc_container(path) / 1024  # Size in KB
            size = f'{size:.2f}'
        else:
            size = 'location not found'
    except pkg_resources.DistributionNotFound:
        size = 'not installed'
    return f'{package:<20}{version:<15}{size:<15}\n'

# Main Function: Print System & Package Information
def print_system_and_package_info(specified_packages: List[str], write_to_file: bool = True) -> None:
    '''
    Write or print system and specified package information along with specified packages only.
    Parameters:
        specified_packages: List[str] - List of packages to check.
        write_to_file: bool - If True, write all information to 'sys-versions.txt'; if False, check and print specified packages only.
    '''
    output = []

    # Check Operating System & Platform
    output.append('System Information: \n\n')
    output.append(f'Operating System: {platform.system()} {platform.release()}\n')
    output.append(f'No. of Processors: {mp.cpu_count()}\n')

    # Check Python Platform & Version
    output.append(f'Python Platform: {platform.platform()}\n')
    output.append(f'Python Version: {sys.version}\n\n')

    # Check conda and pip versions
    try:
        output.append(f'Conda Version: {run_command(["conda", "--version"])}\n')
    except FileNotFoundError:
        output.append('Conda is not installed or not found in PATH.\n')
    try:
        output.append(f'Pip Version: {run_command(["pip", "--version"])}\n')
    except FileNotFoundError:
        output.append('Pip is not installed or not found in PATH.\n')

    # Check specified packages and format as a table
    output.append('\n\nSpecified Packages:\n\n')
    output.append(f'{"Package":<20}{"Version":<15}{"Size (KB)":<15}\n')
    output.append(f'{"-"*50}\n')
    
    for package in specified_packages:
        if package.strip():
            output.append(get_package_info(package))
        else:
            output.append('\n')  # Add empty line for empty package name

    # Conditional logic based on write_to_file
    if write_to_file:
        # Write all information, including installed packages, to file
        output.append('\nInstalled Packages:\n\n')
        output.append(f'{"Package":<20}{"Version":<15}{"Size (KB)":<15}\n')
        output.append(f'{"-"*50}\n')

        for dist in pkg_resources.working_set:
            output.append(get_package_info(dist.project_name))

        # Write output to file
        with open('sys-versions.txt', 'w') as f:
            f.writelines(output)
        print('System, specified package information, and installed packages have been written to sys-versions.txt')

    else:
        # Print specified package information only
        for line in output:
            print(line, end='')