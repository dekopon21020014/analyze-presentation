#!/bin/bash
docker build -t voice .

# if you want to gpu
# docker build -t voice:cuda -f Dockerfile.gpu .