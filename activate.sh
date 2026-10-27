#!/bin/bash

# to enter in the env, you run this command:
# . ./activate.sh

echo "Activating PC + pySLAM environment..."

# 1. Add pySLAM to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/home/igrilex/ASP-SLAM/pyslam"

# 2. Activate the pySLAM venv (this also sets the (pyslam) prompt)
. /home/igrilex/ASP-SLAM/pyslam/pyenv-activate.sh

echo "PC project + pySLAM venv activated!"