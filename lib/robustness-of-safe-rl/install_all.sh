# setup safety gym
cd envs/Bullet-Safety-Gym
pip install -e .
cd ..
pip install -r requirement.txt
pip install -e .

#SLGD optimizer for MAD
pip install git+https://github.com/MFreidank/pysgmcmc@pytorch

echo "Please install pytorch manually to finish the env setup."
