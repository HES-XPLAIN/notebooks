FROM jupyter/base-notebook:python-3.10

RUN apt-get update
RUN apt-get install 

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch and torchvision
RUN pip3 install torch torchvision -f https://download.pytorch.org/whl/cu118/torch_stable.html
# torchaudio
# RUN pip3 install 'torch==2.0.1+cpu' 'torchvision==0.15.2+cpu' --index-url https://download.pytorch.org/whl/cpu 
# 'torchaudio==2.0.2+cpu'
