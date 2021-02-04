#!/bin/bash

# This is an example call to train the UNet architecture with:
#      * 'pot' loss function
#      * 10 epochs
#      * loss regularization term = 10
#      * batch size = 10
python main.py unet -e 10 -a 10 -b 2
