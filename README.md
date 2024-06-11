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
    && pip install box2d-py==2.3.5 pygame==2.1.2 pyglet==1.5.0 \
    && pip install rtamt psy_taliro
```

Configure the bullet_robust environment, this environment is used to run the PyBullet case studies:
```
conda activate bullet_robust
pip install -e . \
    && pip install -e ./lib/Bullet-Safety-Gym \
    && pip install -e ./lib/robustness-of-safe-rl \
    && pip install -r ./lib/robustness-of-safe-rl/requirement.txt \
    && pip install git+https://github.com/MFreidank/pysgmcmc@pytorch \
    && pip install rtamt psy_taliro
```

To run the Matlab Simulink environments, either environment should work. However, it should have Matlab installed. We tested against **Matlab R2022b**.

## Folder structure
- `robustness`: the Python package for our robustness evaluation tool.
- `data`: the saved restuls for our benchmark.
- `gifs`: the saved diagrams for our benchmark results.
- `lib`: the dependencies for using Breach, Bullet-Safety-Gym, and safe-rl package.
- `models`: trained RL models and Matlab Simulink models for our environments.
- `scripts`: the run scripts for running our benchmark problems.
  - `baselines`: the run scripts for running the one-layer search baselines.
  - `testing`: the run scripts for running the CMA-ES search.
  - `heuristics`: the run scripts for running the CMA-ES+Heuristic search.
- `tests`: a set of tests to validate basic functionalities of our tool.

## Run the benchmark
To run a particular problem, you can do, for example:
```
conda activate gym_robust
python scripts/testing/cartpole_dqn.py
```
This script runs the cartpole+dqn benchmark problem with CMA-ES. The results will be stored in `data/cartpole-dqn/cma` and the figures are in `gifs/cartple-dqn`. By default, the script will load the saved data. So to do a fresh run, you can delete the stored data in the `data/cartpole-dqn` folder.

Running other benchmark problems in other modes is similar, just to find the corresponding run scripts under `scripts`.

## Run the tests
However, the benchmark problems take too long to run. To validate the basic functionality of our tool, you can run the test scripts under `tests`.

### CMA-ES test
```
conda activate gym_robust
python tests/test_cartpole_dqn_cma.py
```

### CMA-ES+Heuristic test
```
conda activate gym_robust
python tests/test_cartpole_dqn_cma_heuristics.py
```

### Breach one-layer baseline test
```
conda activate gym_robust
python tests/test_WTK_T_baseline.py
```

### PsyTaLiRo one-layer baseline test
```
conda activate gym_robust
python tests/test_cartpole_dqn_baseline.py
```
