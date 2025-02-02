import subprocess
import sys
import os

# Helper Function: Run shell commands
def run_command(command: str) -> None:
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f'Command failed: {command}')
        sys.exit(result.returncode)

# Main Function: Install dependencies
def main() -> None:
    # PYTHON 3.11.4
    commands = [
        # '%pip uninstall libraryXYZ -y' # Uninstall a package if necessary
        'pip install --upgrade pip',
        'pip install "cython<3.0.0" wheel',
        'pip install "pyyaml==5.4.1" --no-build-isolation',
        'pip install lxml contextlib2',
        'pip install --no-binary :all: --no-use-pep517 numpy',
        'pip install typing-extensions==4.1',        
        'pip install imageio',
        'pip install --no-cache-dir pillow',
        'pip install datasets huggingface-hub librosa wordcloud',
        'pip install --upgrade evaluate jiwer',
        'pip install seaborn matplotlib',
        'pip install transformers',
        'pip install tf-keras',
        'pip uninstall -y tensorflow tensorflow-macos tensorflow-metal keras',
        'pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0',
        'pip install tensorflow==2.16.1 tensorflow-metal==1.1.0 keras==3.0.0'
        'pip install pydub'       
    ]
    
    # Platform-specific commands
    # Check if the system is macOS, Linux or Windows
    if os.name == 'posix':
        if 'Darwin' in os.uname().sysname:  # macOS
            print('Detected macOS system')
            # commands.append('brew install libjpeg')
            # commands.append('brew install protobuf')
            # commands.append('brew install ffmpeg') # pydub relies on ffprobe to process audio files like MP3
            # commands.append('pip install tensorflow-macos==2.13.0 tensorflow-metal') # ==1.0.1
        else:  # Linux
            print('Detected Linux system')
            commands.append('sudo apt-get install libjpeg-dev')
    
    elif os.name == 'nt':  # Windows
        print('The Windows builds of PIL include libjpeg by default.')

    # Execute each command in the list
    for command in commands:
        print(f'Running: {command}')
        run_command(command)

if __name__ == '__main__':
    main()