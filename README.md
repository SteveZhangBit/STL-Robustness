# STL-Robustness

## Install
The implementation is tested based on Ubuntu 20.04. Run the following command to install required dependencies and the Python libraries. We use conda to create virtual environments since some of the case studies require different dependencies.

```
sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gcc \
        g++ \
        libboost-all-dev \
        make \
        cmake \
        antlr4 \
    && curl -fsSL -o miniconda-install.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh \
    && bash miniconda-install.sh -b -p /miniconda \
    && /miniconda/bin/conda create -n gym_robust -y python=3.8 \
    && /miniconda/bin/conda create -n bullet_robust -y python=3.8 \
    && /miniconda/bin/conda init bash
```

Under the root of this project, we need to create two conda virtual environments.

Configure the gym_robust environment, this environment is used to run the OpenAI-Gym case studies:
```
conda activate gym_robust
pip install -e . \
    && pip install swig gym==0.21.0 stable-baselines3==1.6.2 \
    && pip install box2d-py==2.3.5 pygame==2.1.2 pyglet==1.5.0
```

Configure the bullet_robust environment, this environment is used to run the PyBullet case studies:
```
conda activate bullet_robust
pip install -e . \
    && pip install -e ./lib/Bullet-Safety-Gym \
    && pip install -e ./lib/robustness-of-safe-rl \
    && pip install -r ./lib/robustness-of-safe-rl/requirement.txt \
    && pip install git+https://github.com/MFreidank/pysgmcmc@pytorch
```

To run the Matlab Simulink environments, either environment should work. However, it should have Matlab installed. We tested against **Matlab R2022b**.

## Run the benchmark
Use the scripts under `scripts/testing` to run all the benchmark. For example to run the gym case studies:
```
conda activate gym_robust
bash scripts/testing/run_all_gym.sh
```

You can also run an individual case study script, for example:
```
conda activate gym_robust
python scripts/testing/cartpole_pid.py
```
