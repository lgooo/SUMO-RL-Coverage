#!/bin/bash

cd ~/sumo
. bin/activate
cd ~/workspace/repos/SUMO-RL-Coverage/sumo_gym
python $@
