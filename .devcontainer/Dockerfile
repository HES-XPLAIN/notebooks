FROM docker.io/jupyter/base-notebook:latest

USER root

RUN apt-get update
RUN apt-get install python3-dev
RUN apt-get install build-essential -y

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision -f https://download.pytorch.org/whl/cu118/torch_stable.html
# torchaudio
# RUN pip3 install 'torch==2.0.1+cpu' 'torchvision==0.15.2+cpu' --index-url https://download.pytorch.org/whl/cpu 
# 'torchaudio==2.0.2+cpu'


# Generally, Dev Container Features assume that the non-root user (in this case jovyan)
# is in a group with the same name (in this case jovyan). So we must first make that so.
RUN groupadd jovyan \
    && usermod -g jovyan -a -G users jovyan

USER jovyan
