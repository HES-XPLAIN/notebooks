FROM ghcr.io/hes-xplain/docker-notebook-base:latest

USER root

# Set environment variables
ARG USER=xplain
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV USER ${USER}
ENV HOME /home/${USER}

# Make sure the contents of our repo is in $HOME
COPY --chown="${USER_UID}:${USER_GID}" use_case_sport_classification/ $HOME

# Set working directory
WORKDIR ${HOME}

# Download external files
RUN curl -OL https://huggingface.co/HES-XPLAIN/sport_classification/resolve/main/FineTunedEfficientNet_30epochs.pth -o models/saved_models/FineTunedEfficientNet_30epochs.pth
RUN curl -OL https://huggingface.co/HES-XPLAIN/sport_classification/resolve/main/VGGFineTuned.pth -o models/saved_models/VGGFineTuned.pth

ENV JUPYTER_PORT=8888
EXPOSE ${JUPYTER_PORT}

USER ${USER}

CMD ["jupyter", "lab", "--ip",  "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

