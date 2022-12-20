On the Robustness of Safe RL under Observational Pertubrations
==================================
## Environment Setup
### System requirements
- Tested in Ubuntu 20.04, should be fine with Ubuntu 18.04
- Recommend to use [Anaconda3](https://docs.anaconda.com/anaconda/install/) for python env management

### Anaconda Python env setup
Back to the repo root folder, **activate a python 3.6+ virtual anaconda env**, and then run
```
cd ../.. && bash install_all.sh
```
It will install the `bullet_safety_gym` and this repo's python package dependencies that are listed in `requirement.txt`. Then install pytorch based on your platform, see [tutorial here](https://pytorch.org/get-started/locally/).

Back to the repo root folder, **activate a python 3.6+ virtual anaconda env**, and then run
```
pip install -r requirement.txt
pip install -e .
```

Then install `pytorch` manually based on your system configurations to finish the env setup, see [instructions here](https://pytorch.org/get-started/locally/).

The experiments (Circle, Run) are conducted in BulletSafetyGym, so if you want to try those environments, install them with following commands:
```
cd envs/Bullet-Safety-Gym
pip install -e .
```
Note that the BulletSafetyGym is modified based on the original one. The major modification is the simulation step where we increase it to reduce the total training time without sacrifacing too much accuracy. 

The MAD attacker requires pysgmcmc library for optimization. Install it by:
```
pip install git+https://github.com/MFreidank/pysgmcmc@pytorch
```

### Structure
The structure of this repo is as follows:
```
Robust safe RL libraries
├── rsrl  # package folder
│   ├── policy # core algorithm implementation
│   ├── ├── model # stores the actor critic model architecture
│   ├── ├── policy_name # algorithms implementation
│   ├── util # logger and pytorch utils
│   ├── runner.py # training logic of the algorithms
│   ├── evaluator.py # evaluation logic of trained agents
├── script  # stores the training scripts.
│   ├── config # stores some configs of the env and policy
│   ├── run.py # launch a single experiment
│   ├── eval.py # evaluate script of trained agents
├── data # stores experiment results
├── misc # stores experimental params
```

### How to run experiments

To run a single experiment:
```
python script/run.py --rs_mode vanilla --policy robust_ppo
```


To evaluate a trained model, run:
```
python script/eval.py -d path_to_model
```

The pretrained model is available at [here](https://drive.google.com/file/d/1DLlRU3w-YPhCYWyARPbbEXI7PT7P_Qzd/view?usp=share_link).

The complete hyper-parameters can be found in `script/config/config_robust_ppo.yaml`. In particular, each algorithm has different robust training modes, which are specified by the `rs_mode` parameter. We detail the modes as follows.

- For PPO-lagrangian and FOCOPS series: the modes are `vanilla, kl, klmc, klmr, mc, mcv, mr, mrv, mad, uniform, gaussian`. The proposed adversarial training methods correspond to the `mc, mcv, mr, mrv` modes, where adding `v` in the suffix means using half vanilla samples for training. The vanilla ppo-lagrangian method is the `vanilla` mode, and the SA-PPOL method with the original MAD attacker is the `kl` mode, the SA-PPOL method with the MC and MR attackers are `klmc` and `klmr`. `mad, uniform, gaussian` are adversarial training under the MAD attacker and random noises correspondingly.

- For CVPO and SAC series: the modes are `random, vanilla_adv, vanilla`, where `vanilla` is pure safe RL method without adversarial training. For online adversarial training by attacking the behavior agents with Gaussian noise or the MC attacker, use `random` and `vanilla_adv` respectively.



### Reference:
Part of the code is based on several public repos:
* https://github.com/SvenGronauer/Bullet-Safety-Gym
* https://github.com/openai/safety-gym
* https://github.com/openai/spinningup
* https://github.com/daisatojp/mpo
* https://github.com/liuzuxin/cvpo-safe-rl
