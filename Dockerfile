FROM jupyter/base-notebook:python-3.10

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        libglib2.0-0
# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip3 install 'torch==2.0.1' 'torchvision==0.15.2' 'torchaudio==2.0.2' --index-url https://download.pytorch.org/whl/cpu

# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT [ "python3" ]
