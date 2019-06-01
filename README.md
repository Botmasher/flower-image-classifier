# AI Programming with Python Project

A neural network that learns to predict . I developed this code as part of a multi-course program about AI Programming with Python. The project contained starter license text, `assets/`, `flowers/`, `cat_to_name.json` and `workspace-utils` and original README text below. All else is my addition. The core of the model lives in `train.py` and `predict.py`.

## Command Line Interface

Running of `train.py` and `predict.py` from the terminal is enhanced with a simple CLI. Required and optional arguments are easily found within the `parse_args` methods in both files.

Example use:
- Training: `python3 train.py flowers --savedir checkpoint.pt --arch densenet --learning_rate 0.004 --epochs 8 --gpu`
- Predicting: `python3 predict.py assets/test/10/image_07475.jpg checkpoint.pt --top_k 3`

## Final Project Description from Udacity

(_Original README body text below._)

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.
