# End-to-End-Regression-Modelling-Task
Modelling an end to end linear regression task

This project trains a single fully-connected layer to fit a linear regression model with 2 features.

# Linear Regression architecture

![](https://github.com/joshir199/End-to-End-Regression-Modelling-Task/blob/main/model_architecture.png)



# Training and Evaluating script:
```bash
usage: main.py [-h] [--seed S] [--outf OUTF] [--ckpf CKPF] [--degree P]
               [--batch-size N] [--train] [--evaluate]

Pytorch Linear Regression

optional arguments:
  -h, --help      show this help message and exit
  --seed S        random seed (default: 1)
  --outf OUTF     folder to output images and model checkpoints
  --ckpf CKPF     path to model checkpoint file (to continue training)
  --degree P      polynomial degree to learn(default: 4)
  --batch-size N  input batch size for training (default: 32)
  --train         training a fully connected layer
  --evaluate      Evaluate a [pre]trained model from a random tensor.
```

# Linear Regression model graph

![](https://github.com/joshir199/End-to-End-Regression-Modelling-Task/blob/main/Linear%20regression.png)

# Training
This project will automatically create a dataset with 2 features and one output value.

Optimizer used in the project : SGD with learning rate 0.1

Here's the commands to training, Please run the following commands by providing appropriate value for mentioned parameters.

full_path : full directory path to this folder where model weights will be saved after training
```bash
$ python main.py --train --seed 3 --outf "full_path/output"
```

# Training Loss Curve
![](https://github.com/joshir199/End-to-End-Regression-Modelling-Task/blob/main/Loss_curve.png)


# Evaluating
Here's the commands to evaluating the model with new data points, Please run the following commands by providing appropriate value for mentioned parameters.

full_path : full directory path to the folder where model weights are stored after training
```bash
$ python main.py --evaluate --seed 3 --ckpf "full_path/output"
```


