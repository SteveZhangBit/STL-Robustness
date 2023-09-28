FROM ubuntu:20.04

ENV TZ=America/New_York \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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

# Compile RTAMT
# RUN git clone https://github.com/nickovic/rtamt.git \
#     && cd /rtamt/rtamt \
#     && mkdir build \
#     && cd build \
#     && cmake -DPythonVersion=3 ../ \
#     && make

# Install conda environments
ENV PATH="${PATH}:/miniconda/bin"
RUN git clone https://github.com/SteveZhangBit/STL-Robustness.git

WORKDIR /STL-Robustness
SHELL ["conda", "run", "--no-capture-output", "-n", "gym_robust", "/bin/bash", "-c"]
RUN pip install -e . \
    && pip install swig gym==0.21.0 stable-baselines3==1.6.2 \
    && pip install box2d-py==2.3.5 pygame==2.1.2 pyglet==1.5.0
    # && cd /rtamt \
    # && pip install . \
    # && pip install psy_taliro \
    # && pip install statsmodels

SHELL ["conda", "run", "--no-capture-output", "-n", "bullet_robust", "/bin/bash", "-c"]
RUN pip install -e . \
    && pip install -e ./lib/Bullet-Safety-Gym \
    && pip install -e ./lib/robustness-of-safe-rl \
    && pip install -r ./lib/robustness-of-safe-rl/requirement.txt \
    && pip install git+https://github.com/MFreidank/pysgmcmc@pytorch
    # && cd /rtamt \
    # && pip install . \
    # && pip install psy_taliro \
    # && pip install statsmodels

CMD ["bash"]