#!/bin/bash
docker run --rm -it -v ./src:/app -p 8000:8000 presentation-analysis

# if you want to use gpu
#docker run --rm --runtime=nvidia --gpus all -it -v ./src:/app -p 8000:8000 presentation-analysis:cuda