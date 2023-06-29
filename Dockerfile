FROM jupyter/base-notebook:python-3.10

# Set environment variables
ARG NB_USER=jovyan
ARG NB_UID="1000"
ARG NB_GID="100"

ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         python3-pip \
#         python3-dev \
#         libglib2.0-0

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
