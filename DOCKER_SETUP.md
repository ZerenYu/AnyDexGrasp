# AnyDexGrasp Docker Setup

This document provides instructions for setting up AnyDexGrasp in a Docker container for both training and inference.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Setup Instructions

1. Clone the AnyDexGrasp repository:
```bash
git clone https://github.com/your-repo/AnyDexGrasp.git
cd AnyDexGrasp
```

2. Build the Docker image:
```bash
docker-compose build
```

3. Create a `logs` directory to store models and data:
```bash
mkdir -p logs/data/representation_model/graspnet_v1_newformat/
```

4. Download model weights and data from [GoogleDrive](https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo) and place them in the `logs/` directory. Extract zip files in `logs/data/representation_model/graspnet_v1_newformat/`.

5. Download the data from [Graspnet](https://graspnet.net/datasets.html) and extract it to `logs/data/representation_model/graspnet_v1_newformat/`.

## Running the Container

Start the container:
```bash
docker-compose up -d
```

Access the container:
```bash
docker exec -it anydexgrasp bash
```

## Usage

### Generating the STL file for dexterous hand
```bash
python command_generate_mesh_file.sh
```

### Training
```bash
sh command_train_representation.sh  # Representation model
sh command_train_multifinger_decision.sh  # Decision model
```

### Collecting data
```bash
python realsense.py
sh command_collect_multifinger_grasp_data.sh
```

### Robot grasp
```bash
python realsense.py
sh command_robot_multifinger_grasp.sh
```

## Stopping the Container

```bash
docker-compose down
```

## Troubleshooting

### Hardware Access
If you need to access hardware devices (like cameras or robot hardware), you may need to add additional volume mounts in the `docker-compose.yml` file:

```yaml
volumes:
  - /dev:/dev
```

### Display Issues
For GUI applications, ensure X11 forwarding is properly set up:

```bash
xhost +local:docker
```

### CUDA Issues
If you encounter CUDA-related errors, verify that the NVIDIA drivers on your host machine are compatible with CUDA 11.7.

## Notes

- The container is set up to use GPU acceleration if available.
- All code changes are preserved between container restarts because the repository is mounted as a volume.
- Training results and models are stored in the `logs` directory, which is also mounted as a volume. 