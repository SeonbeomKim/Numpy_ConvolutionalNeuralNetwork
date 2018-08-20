import numpy as np
import layer as nn
import common.initializer as init
import common.model as net
import common.util as util
import common.optimizer as optimizer
from tensorflow.examples.tutorials.mnist import input_data #for MNIST dataset

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = np.hstack((mnist.train.images, mnist.train.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.
vali_set = np.hstack((mnist.validation.images, mnist.validation.labels))
test_set = np.hstack((mnist.test.images, mnist.test.labels))

#train_set = train_set[:128]
#vali_set = vali_set[:128]
#test_set = test_set[:128]

def train(model, data):
	batch_size = 128
	loss = 0
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784].reshape(-1, 28, 28, 1)
		target_ = batch[:, 784:]

		logits = model.forward(input_, is_train=True)
		train_loss = model.backward(logits, target_)
		loss += train_loss
	return loss/len(data)


def validation(model, data):
	batch_size = 128
	loss = 0
	
	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784].reshape(-1, 28, 28, 1)
		target_ = batch[:, 784:]
	
		logits = model.forward(input_, is_train=False)
		vali_loss = model.get_loss(logits, target_)
		loss += vali_loss
	return loss/len(data)


def test(model, data):
	batch_size = 128
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784].reshape(-1, 28, 28, 1)
		target_ = batch[:, 784:]

		logits = model.forward(input_, is_train=False)
		check = model.correct(logits, target_, axis=1)
		correct += check
	return correct/len(data)


def run(model, train_set, vali_set, test_set):
	for epoch in range(1, 300):
		train_loss = train(model, train_set)
		vali_loss = validation(model, vali_set)
		accuracy = test(model, test_set)

		print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy)


lr = 0.01

model = net.model(optimizer.Adam(lr=lr)) # 30 66
#model = net.model(optimizer.GradientDescent(lr=lr))  #30번에 32퍼 학,검,테 데이터셋 128개일때 

model.add(nn.conv2d(filters=32, kernel_size=[3,3], strides=[1,1], w_init=init.he))
model.add(nn.relu())
model.add(nn.maxpool2d(kernel_size=[2,2], strides=[2,2]))
model.add(nn.dropout(0.6))

model.add(nn.conv2d(filters=64, kernel_size=[3,3], strides=[1,1], w_init=init.he))
model.add(nn.relu())
model.add(nn.maxpool2d(kernel_size=[2,2], strides=[2,2]))
model.add(nn.dropout(0.6))

model.add(nn.conv2d(filters=128, kernel_size=[3,3], strides=[1,1], w_init=init.he))
model.add(nn.relu())
model.add(nn.maxpool2d(kernel_size=[2,2], strides=[2,2]))
model.add(nn.dropout(0.6))

model.add(nn.flatten())
model.add(nn.affine(out_dim=10, w_init=init.he))

model.add_loss(nn.softmax_cross_entropy_with_logits())


run(model, train_set, vali_set, test_set)
