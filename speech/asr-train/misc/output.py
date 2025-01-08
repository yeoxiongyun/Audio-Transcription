from IPython.core.magic import register_cell_magic
import os

@register_cell_magic
def capture_to_file(line: str, cell: str) -> None:
    '''
    Cell magic to capture output and write to a specified file with visual separation between cells.
    
    Usage:
    %%capture_to_file output.txt 1
    <cell code here>
    This will write the output of the cell into `output.txt` with visual markers.
    
    Parameters:
    - line: Should contain the file name and optional cell ID (e.g., 'output.txt 1')
    - cell: The code block to execute and capture output from
    '''
    # Parse the file name and optional cell ID from the line input
    parts = line.strip().split()
    output_file = parts[0] if len(parts) > 0 else 'output.txt'
    cell_id = parts[1] if len(parts) > 1 else 'unknown'
    
    # Standard 80-character separation line
    separator_line = '#' * 80
    start_marker = f'Cell {cell_id} {separator_line}\n'
    end_marker = f'{separator_line}\n'
    
    # Capture the output of the cell
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr
    import sys
    
    # Create a string buffer to capture stdout and stderr
    stdout = StringIO()
    stderr = StringIO()
    
    # Redirect stdout and stderr to capture the output of the cell
    with redirect_stdout(stdout), redirect_stderr(stderr):
        # Execute the cell code within the global and local namespace
        exec(cell, globals(), locals())  # Pass global and local variables
    
    # Get the captured output
    output = stdout.getvalue() + stderr.getvalue()
    
    # Read the existing content of the output file, if it exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            file_content = f.readlines()
    else:
        file_content = []
    
    # Locate the start and end points of the current cell's output
    start_index = None
    end_index = None
    for i, line in enumerate(file_content):
        if line == start_marker:
            start_index = i
        if line == end_marker:
            end_index = i
    
    # If found, replace the old output with the new one
    if start_index is not None and end_index is not None:
        new_file_content = file_content[:start_index + 1] + [output + '\n'] + file_content[end_index:]
    else:
        # If not found, append new output at the end
        new_file_content = file_content + [start_marker, output + '\n', end_marker]
    
    # Write the updated content back to the file
    with open(output_file, 'w') as f:
        f.writelines(new_file_content)
