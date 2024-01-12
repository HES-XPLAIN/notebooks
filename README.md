# HES-XPLAIN notebooks

Public notebooks for the [hes-xplain.github.io](https://hes-xplain.github.io/) website.

The current notebooks include:

* **Class Activation Maps**: Sport Image Classification
* **Rules extraction**: Sport Image Classification
* **Activation-Maximizatio**: WIP

## Docker image

A Docker image containing the HES-XPLAIN notebooks is provided.
The image is built upon the [docker-notebook-base](https://github.com/HES-XPLAIN/docker-notebook-base)
which include all the libraries needed to run the notebooks.

### Run locally

If you simply want to use the notebooks locally, you can use the provided Compose file.

* Install and launch [Docker Desktop](https://www.docker.com/).
* Download the `docker-compose.yml` file.
* Deploy the Docker image with `docker compose up`.

## Adding a new notebook

When adding a new notebook and its related code, ensure the related folder is
also added to the `Dockerfile`, i.e.:

```
# Make sure the contents of our repo is in $HOME
COPY --chown="${USER_UID}:${USER_GID}" use_case_sport_classification/ $HOME
```

> **Important**
> If new dependencies to run the notebook experiments are needed, they should be
> included in the [docker-notebook-base](https://github.com/HES-XPLAIN/docker-notebook-base) image.
>
> A new base image needs to be then generated. Refer to the related [README](https://github.com/HES-XPLAIN/docker-notebook-base/blob/main/README.md).

> **Note**
> The current list of included high-level dependencies is visible here: [requirements.txt](https://github.com/HES-XPLAIN/docker-notebook-base/blob/main/requirements.txt)

## Build

### authenticate (build from ghcr.io)

A [Personal Authentication Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
with the `read repository` permission needs to be generated.

```shell
PAT=abcdef123456789
echo $PAT | docker login -u <username> ghcr.io --password-stdin
```

This is necessary to pull the docker-notebook-base image that is hosted on the GitHub Container registry.

### build

```
docker buildx build -t notebooks .
```

## Run

### run and launch jupyter


To use GPU support (CUDA), ensure the [nVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed.

```
docker run --rm -p 8080:8080 -p 8888:8888 -it --gpus all notebooks
```

To run the image with CPU support only:

```
docker run --rm -p 8080:8080 -p 8888:8888 -it notebooks
```

### run and override entry point

```
docker run --rm -it --entrypoint /bin/bash notebooks
```

## Release (manual)

### authenticate (push to ghcr.io)

A [Personal Authentication Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
with the `read repository` permission needs to be generated.

```shell
PAT=abcdef123456789
echo $PAT | docker login -u <username> ghcr.io --password-stdin
```

### tag and push image to registry

```shell
docker image tag notebooks:latest ghcr.io/hes-xplain/notebooks:latest
docker push ghcr.io/hes-xplain/notebooks:latest
```

If `ghcr.io` is omitted, the registry used will be [Docker Hub](https://hub.docker.com/).

## Release

To publish the package on the GitHub Packages registry, see [RELEASE](RELEASE.md).
