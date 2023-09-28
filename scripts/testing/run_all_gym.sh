#!/bin/bash

# Run the cartpole_dqn env
python scripts/testing/cartpole_dqn.py
# Run the cartpole_pid env
python scripts/testing/cartpole_pid.py
# Run the cartpole4_pid env
python scripts/testing/cartpole4_pid.py
# Run the cartpole4_dqn env
python scripts/testing/cartpole4_dqn.py

# Run the lunar_lander_lqr env
python scripts/testing/lunar_lander_lqr.py
# Run the lunar_lander_ppo env
python scripts/testing/lunar_lander_ppo.py
# Run the lunar_lander3_lqr env
python scripts/testing/lunar_lander3_lqr.py
# Run the lunar_lander3_ppo env
python scripts/testing/lunar_lander3_ppo.py

exit 0
