"""Simple Neural Networks to classify the dating dataset"""
import sys
import time

import pandas as pd
import numpy as np

np.random.seed(0)

def activation(z, derivative=False):
    if derivative:
        return activation(z) * (1 - activation(z))
    else:
        return 1 / (1 + np.exp(-z))

def loss(y_true, y_pred):
    loss = (1./(2*len(y_pred))) * np.sum((y_true - y_pred) ** 2)
    return loss

def forward_pass(input, weights, bias):
	a = input
	pre_activations = []
	activations = [a]
	for w, b in zip(weights, bias):
		z = np.dot(w, a) + b
		a  = activation(z)
		pre_activations.append(z)
		activations.append(a)
	return a, pre_activations, activations

def compute_deltas(pre_activations, y_true, y_pred, layers, weights):
	delta_l = (y_pred - y_true) * activation(pre_activations[-1], derivative=True)
	deltas = [0] * (len(layers)-1)
	deltas[-1] = delta_l
	for l in range(len(deltas)-2, -1, -1):
		delta = np.dot(weights[l + 1].T, deltas[l + 1]) * activation(pre_activations[l], derivative=True) 
		deltas[l] = delta
	return deltas

def backward_pass(deltas, pre_activations, activations, size):
	dW = []
	db = []
	deltas = [0] + deltas
	for l in range(1, len(size)):
		dW_l = np.dot(deltas[l], activations[l-1].T) 
		db_l = deltas[l]
		dW.append(dW_l)
		db.append(np.expand_dims(db_l.mean(axis=1), 1))
	return dW, db

def train_nn(x_train, y_train, weights, bias, layers, batch_size = 32, num_epochs = 500, learning_rate = 0.1):
	for _ in range(num_epochs):
		if x_train.shape[1] % batch_size == 0:
			n_batches = int(x_train.shape[1] / batch_size)
		else:
			n_batches = int(x_train.shape[1] / batch_size ) - 1

		batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(n_batches)]
		batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(n_batches)]

		dW = [np.zeros(w.shape) for w in weights]
		db = [np.zeros(b.shape) for b in bias]
		
		for batch_x, batch_y in zip(batches_x, batches_y):
			batch_y_pred, pre_activations, activations = forward_pass(batch_x, weights, bias)
			deltas = compute_deltas(pre_activations, batch_y, batch_y_pred, layers, weights)
			dW_, db_ = backward_pass(deltas, pre_activations, activations, layers)
			for i, (dw_i, db_i) in enumerate(zip(dW_, db_)):
				dW[i] += dw_i / batch_size
				db[i] += db_i / batch_size

		# weight update
		for i, (dw_e, db_e) in enumerate(zip(dW, db)):
			weights[i] -= learning_rate * dw_e
			bias[i] -= learning_rate * db_e

	return weights, bias

def predict(a, weights, bias):
	for w, b in zip(weights, bias):
		z = np.dot(w, a) + b
		a = activation(z)
	return np.where(a > 0.5, 1, 0)

def accuracy(original_labels, predicted_labels):
    count = 0
    total_num = len(original_labels)
    for idx in range(total_num):
        if original_labels[idx] == predicted_labels[idx]:
            count += 1
    return float(count)/total_num

def neural_net(train_set, test_set):
	layers = [49, 10, 5, 1]
	
	weights = [ np.random.randn(layers[i], layers[i-1]) * np.sqrt(1 / layers[i-1]) for i in range(1, len(layers)) ]
	bias = [ np.random.rand(n, 1) for n in layers[1:] ]
	
	x_train = train_set[:,:-1]
	y_train = train_set[:,-1].reshape(-1, 1)
	weights, bias = train_nn(x_train.T, y_train.T, weights, bias, layers, \
		batch_size = 64, num_epochs = 3000, learning_rate = 0.3)
	y_train_pred = predict(x_train.T, weights, bias)

	x_test = test_set[:,:-1]
	y_test = test_set[:,-1].reshape(-1, 1)
	y_test_pred = predict(x_test.T, weights, bias)
	
	train_accuracy = accuracy(y_train_pred[0], y_train)
	test_accuracy = accuracy(y_test_pred[0], y_test)

	return train_accuracy, test_accuracy

if __name__ == '__main__':
	st = time.time()
	training_data_filename = sys.argv[1]
	test_data_file_name = sys.argv[2]
	train_df = pd.read_csv(str(training_data_filename))
	test_df = pd.read_csv(str(test_data_file_name))

	train_set = train_df.to_numpy()
	test_set = test_df.to_numpy()

	train_acc, test_acc = neural_net(train_set, test_set)
	print(f'Training Accuracy NN: {train_acc}')
	print(f'Testing Accuracy NN: {test_acc}')
	print(f'Total Time Elapsed: {time.time()-st} seconds')