# Hash NeRF

This is an implementation of HasHNerf in Pytorch. The fundational paper is [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf) and this implementation is based on [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

The code is based on the code of [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch) and [Instant ngp](https://github.com/NVlabs/instant-ngp).

## Configuration

### Dependencies

The dependencies for this project are managed by [Poetry](https://python-poetry.org). To install them, run

```bash
poetry install
```

Some of the dependencies are:
- Pytorch 2.0
- Python 3.10

### Docker

A Dockerfile is provided to run the code in a container. To build the image, run

```bash
./build_docker_image.sh
```

The image name is `$HOSTNAME/nerf`. To run the container, run

```bash
./docker.sh python -m nerf.main --help
```

### Hardware

This code was developed and tested on the Nvidia 4090 GPU with 24GB of memory.

## Usage

### Dataset

In order to download the dataset, run

```bash
./docker.sh python -m nerf.main dataset download
```

and it will dowloaded under `-./data`.

### Training

In order to train a model, run

```bash
./docker.sh python -m nerf.main train data/nerf_synthetic/lego blender 
```

### Validation and metrics

_TODO_

### Inference

_TODO_
