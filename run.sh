#!/bin/bash

python3 -m trainer.run
cd config/
yq -i '.computers.docker.file_paths.num_trial=4' config.yml
cd ..
python3 -m trainer.run
