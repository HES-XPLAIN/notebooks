FROM jupyter/base-notebook:python-3.10

USER root

# # Set environment variables
ARG NB_USER=jovyan
# ARG NB_UID=1000
ENV USER ${NB_USER}
# ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}
#
# RUN adduser --disabled-password \
#     --gecos "Default user" \
#     --uid ${NB_UID} \
#     ${NB_USER}

RUN apt update -y
RUN apt-get install -y build-essential

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
# USER root
RUN chown -R ${NB_UID} ${HOME}
# USER ${NB_USER}

# Set the working directory
WORKDIR ${HOME}

# Install any python packages you need
# COPY requirements.txt requirements.txt

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir -r requirements.txt



# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip3 install 'torch==2.0.1+cpu' 'torchvision==0.15.2+cpu' --index-url https://download.pytorch.org/whl/cpu
# 'torchaudio==2.0.2+cpu'
#RUN pip3 install 'torch==2.0.1+cpu' 'torchvision==0.15.2+cpu' 'torchaudio==2.0.2+cpu' -f https://download.pytorch.org/whl/torch_stable.html

USER ${NB_USER}

# Set the entrypoint
ENTRYPOINT [ "python3" ]
