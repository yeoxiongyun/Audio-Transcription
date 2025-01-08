# Use the official Python image based on Debian
FROM python:3.11.4-slim
# Use the official Python image based on Alpine
# FROM python:3.11.4-alpine
# Use the official Python image based on Ubuntu
# FROM python:3.11.4

# Run system update and install required system packages
RUN apt-get update && apt-get install -y \
# Install a package that includes references to the most commonly used development tools
    build-essential \  
    # Install development files for the SSL library
    libssl-dev \       
    # Install Foreign Function Interface library for calling C code
    libffi-dev \       
    # Install development files for the compression library
    zlib1g-dev \       
    # Install a helper tool used when compiling applications and libraries
    pkg-config \       
    # Install development files for the HDF5 library used for managing large data
    libhdf5-dev \      
    # Install wget, a utility for non-interactive download of files from the web
    wget \              
    # Install ffmpeg for PyDub audio processing
    ffmpeg \ 
    # Clean up the apt cache to reduce image size
    && apt-get clean \ 
    # Remove the cached list of packages to save space
    && rm -rf /var/lib/apt/lists/*  

# Set the container working directory
# Root directory is /workspace
WORKDIR /workspace/asr

# # Copy the parent directory to the container
# COPY ../ /workspace/

# Copy the asr folder
COPY speech/asr/ /workspace/asr/
# Copy files to leverage caching for dependencies (cv-valid-dev folder & cv-valid-dev.csv file)
COPY speech/common_voice/cv-valid-dev/ /workspace/common_voice/cv-valid-dev/ 
COPY speech/common_voice/cv-valid-dev.csv /workspace/common_voice/

# Install Python packages with specified versions in a single RUN command to reduce layers
RUN pip3 install --no-cache-dir \ 
    -r docker/requirements.txt

# Expose the port the API will run on
EXPOSE 8001

# FOR DEVELOPMENT OR TESTING

# Run Flask application when container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=8001"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=1234", "--no-browser", "--allow-root"]

# FOR PRODUCTION

# Run API when container launches
# CMD ["python", "asr_api.py"]