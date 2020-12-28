# Evaluation-Metrics-PyTorch

This script is designed to compute multiple metrics of model complexity, including the number of model parameters, the size of model memory (measured in MB), the amount of floating point operations (measured in FLOPs w.r.t a predefined input size (e.g., 224x224)), and the average running speed or fps (w.r.t a predefined input size (e.g., 224x224) and GPU/CPU computing devices). 

This script is borrowed heavily from https://github.com/deep-learning-algorithm/Evaluation-Metrics. Thanks for the contributions from this author. 

# Other useful toolkit

- torchstat:                    https://github.com/Swall0w/torchstat 
- flops-counter.pytorch:        https://github.com/sovrasov/flops-counter.pytorch 
- thop:                         https://github.com/Lyken17/pytorch-OpCounter 
