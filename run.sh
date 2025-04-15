#!/bin/bash

python3 -m trainer.run
cd config/
yq -i '.computers.docker.file_paths.CC_wsis=[6, 8, 11, 12, 13, 14, 39, 52, 54, 55]' config.yml
cd ..
python3 -m trainer.run
cd config/
yq -i '.computers.docker.file_paths.CC_wsis=[72, 108, 111, 116, 122, 124, 130, 131, 137, 138]' config.yml
cd ..
python3 -m trainer.run