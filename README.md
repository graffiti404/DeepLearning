# Deep Learning

[TOC]



## Neural Network Part 1: Inside the black box

### What Is Neural Network?

- a neural network consists of nodes and connections between the nodes.

  [^nodes]: include Hidden Layer nodes ,input nodes ,output nodes .

  - the numbers along  each connection represent parameter values that were estimated when this neural network was fit to the data.
  - using **back propagation**(BP) to estimate unknown parameter values.

#### Activation Function

- to build blocks for fitting a squiggle to data.

- type of function

  - ReLU

  - SoftPlus(soft ReLU)

  - sigmoid

#### Hidden Layer

- layer between the input and output nodes.

#### NOTE

- approximately like a big fancy squiggle fitting machine
  - nodes are like neurons
  - connection are sort of like synapses

#### Name

- the parameters that we multiply are called weights-$w$

- the parameters that we add are called biases

  <img src="README/image-20240522152509931.png" alt="image-20240522152509931" style="zoom:25%;" />

### The Chain Rule

e.g. use weight measurements to predict height and to predict shoe size

#### essence of the chain rule

$$
\frac{dSize}{dWeight}=\frac{dSize}{dHeight}\times\frac{dHeight}{dWeight}
$$

- the chain rule applies to the residual sum of squares(残差平方和)

### Gradient Descent
