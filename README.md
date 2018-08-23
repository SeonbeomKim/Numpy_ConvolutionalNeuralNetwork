# Numpy_ConvolutionalNeuralNetwork
Numpy_ConvolutionalNeuralNetwork

Numpy로 구현한 Convolutional Neural Network
Computational graph를 이용한 Backpropagation  

## common/initializer.py, model.py, optimizer,py, util.py
    * AdamOptimizer, GradientDescentOptimizer 등등 구현

## layer.py
    * Numpy로 구현한 Convolutional Neural Network, Fully Connected Neural Network 함수들의 모음
    * Computational graph(계산 그래프) 방식으로 Backpropagation 진행하도록 구현

## cnn.py
    * common폴더와 layer의 function을 사용하여 Train, Validation, Test를 진행.
    * dataset : MNIST  (dataset을 위해 TensorFlow function 사용)
        * from tensorflow.examples.tutorials.mnist import input_data #for MNIST dataset

## Reference
    * Deep Learning from Scratch 서적.
