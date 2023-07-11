FROM docker.io/jupyter/base-notebook:latest

USER root

# Set environment variables
ARG USER=jovyan
ARG UID=1000

ENV USER ${USER}
ENV HOME /home/${USER}

RUN apt-get update -y
RUN apt-get install -y build-essential
# generic GL provider. FIXME: install nVidia version
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

# Make sure the contents of our repo is in $HOME
#COPY . $HOME
COPY requirements.txt $HOME/
COPY use_case_sport_classification/sport_image_classification.ipynb $HOME/
COPY use_case_sport_classification/data/ $HOME/data/
COPY use_case_sport_classification/models/ $HOME/models/
COPY use_case_sport_classification/scripts/ $HOME/scripts/
COPY use_case_sport_classification/README.md $HOME/
RUN chown -R ${UID} ${HOME}

# Set working directory
WORKDIR ${HOME}

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch and torchvision
#RUN python -m pip install 'torch==2.0.1' 'torchvision==0.15.2' -f https://download.pytorch.org/whl/cu118/torch_stable.html
# 'torchaudio==2.0.2'
RUN python -m pip install 'torch==2.0.1+cpu' 'torchvision==0.15.2+cpu' --index-url https://download.pytorch.org/whl/cpu
# 'torchaudio==2.0.2+cpu'

ENV JUPYTER_PORT=8888
EXPOSE ${JUPYTER_PORT}

# Generally, Dev Container Features assume that the non-root user (in this case jovyan)
# is in a group with the same name (in this case jovyan). So we must first make that so.
RUN groupadd ${USER} && usermod -g ${USER} -a -G users ${USER}

USER ${USER}

CMD ["jupyter", "lab", "--ip",  "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

