FROM ghcr.io/hes-xplain/docker-notebook-base:latest

USER root

# Set environment variables
ARG USER=jovyan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV USER=${USER}
ENV HOME=/home/${USER}

# Make sure the contents of our repo is in $HOME
COPY --chown="${USER_UID}:${USER_GID}" use_case_sport_classification/ $HOME

# Set working directory
WORKDIR ${HOME}

ENV JUPYTER_PORT=8888
EXPOSE ${JUPYTER_PORT}

USER ${USER}

CMD ["jupyter", "lab", "--ip",  "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

