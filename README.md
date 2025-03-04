# AnyDexGrasp
Dexterous grasp pose detection network built upon [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

## Requirements
- [Anaconda](https://www.anaconda.com/) with Python 3.8
- PyTorch 1.13 with CUDA 11.7
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5

## Installation
1. Follow MinkowskiEngine [instructions](https://github.com/NVIDIA/MinkowskiEngine#anaconda) to install [Anaconda](https://www.anaconda.com/), cudatoolkit, Pytorch and MinkowskiEngine.**Note that you need ``export MAX_JOBS=2;`` before ``pip install ``due to [this issue](https://github.com/NVIDIA/MinkowskiEngine/issues/228)**

2. Install other requirements from Pip.
```bash
    pip install -r requirements.txt
```

3. Install ``knn`` module.
```bash
    cd knn
    python setup.py install
```

4. Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

5. Install ur toolbox.
```bash
    cd ur_toolbox
    pip install .
    cd python-urx
    pip install .
    pip install -r requirements.txt
```
6. Install Allegro Hand.

   Install Allegro Hand upon [Allegro-Hand-Controller-DIME](https://github.com/NYU-robot-learning/Allegro-Hand-Controller-DIME).

7. Download model weights and data at [GoogleDrive](https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo) and put it under ``logs/`` and extract zip files in ``logs/data/representation_model/graspnet_v1_newformat/`` to ``logs/data/representation_model/graspnet_v1_newformat/``.   
   Download the data from the [Graspnet](https://graspnet.net/datasets.html) web page and extract it to ``logs/data/representation_model/graspnet_v1_newformat/``

## Generating the STL file for dexterous hand.
```bash
    python command_generate_mesh_file.sh
```

## Training
```bash
    sh command_train_representation.sh # Representation model
    sh command_train_decision.sh # Decision model
```


## Collecting data
```bash
    python realsense.py
    sh command_collect_multifinger_grasp_data.sh
```


## Robot grasp
```bash
    python realsense.py
    sh command_robot_multifinger_grasp.sh
```

