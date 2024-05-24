# import some package

import torch
# create tensors to store all the numerical values including the raw data and the values for each weight and bias
import torch.nn as nn
# make the weight and bias tensors part of the neural network
import torch.nn.functional as F
# provide the activation functions
from torch.optim import SGD
# short for stochastic gradient descent to fit the neural network to data
import matplotlib.pyplot as plt
import seaborn as sns


# draw nice looking graphs

# creating a new neural network

class BasicNN(nn.Module):  # inherit from a PyTorch class called module
    def __init__(self):
        super().__init__()
        # set parameter
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        # neural network can take advantage of the accelerated arithmetic and automatic differentiation
        # don't need to optimize this weight,so requires_grad=False
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input):  # connect input, layer and output; making a forward pass through
        # connect top input and ReLU
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        # connect bottom input and ReLU
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output


input_doses = torch.linspace(start=0, end=1, steps=11)

model = BasicNN()

output_values = model(input_doses)

sns.set_style("white")

sns.lineplot(x=input_doses,
             y=output_values,
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')