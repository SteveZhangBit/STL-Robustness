#!/bin/bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Run the ACC_RL env
python scripts/testing/ACC_RL.py
# Run the ACC_T env
python scripts/testing/ACC_T.py
# Run the ACC3_RL env
# python scripts/testing/ACC3_RL.py
# Run the ACC3_T env
python scripts/testing/ACC3_T.py

# Run the AFC_RL env
python scripts/testing/AFC_RL.py
# Run the AFC_T env
python scripts/testing/AFC_T.py

# Run the WTK_RL env
python scripts/testing/WTK_RL.py
# Run the WTK_T env
python scripts/testing/WTK_T.py

exit 0
