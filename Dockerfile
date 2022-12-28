FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gcc \
        g++ \
    && curl -fsSL -o miniconda-install.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh \
    && bash miniconda-install.sh -b -p /miniconda \
    && /miniconda/bin/conda create -n gym_robust -y python=3.8 \
    && /miniconda/bin/conda create -n bullet_robust -y python=3.8 \
    && /miniconda/bin/conda init bash

ENV PATH="${PATH}:/miniconda/bin"
COPY ./ /STL-Robustness
WORKDIR /STL-Robustness

SHELL ["conda", "run", "--no-capture-output", "-n", "gym_robust", "/bin/bash", "-c"]
RUN pip install -e . \
    && pip install swig gym==0.21.0 stable-baselines3==1.6.2 \
    && pip install box2d-py==2.3.5 pygame==2.1.2 pyglet==1.5.0

SHELL ["conda", "run", "--no-capture-output", "-n", "bullet_robust", "/bin/bash", "-c"]
RUN pip install -e . \
    && pip install -e ./lib/Bullet-Safety-Gym \
    && pip install -e ./lib/robustness-of-safe-rl \
    && pip install -r ./lib/robustness-of-safe-rl/requirement.txt \
    && pip install git+https://github.com/MFreidank/pysgmcmc@pytorch

CMD ["bash"]