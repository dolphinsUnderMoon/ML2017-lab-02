from sklearn import datasets as ds
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def sigmoid(input_):
    return 1 / (1 + np.exp(-input_))


def train(x_train, y_train, x_test, y_test, method, iters, test_errors):
    max_iterations = 100
    theta = random.rand(num_features + 1)
    num_test_samples, num_test_features = x_test.shape

    if method == 'sgd':
        lr = 0.01

        for i in range(max_iterations):
            output = sigmoid(np.dot(x_train[i], theta))
            error = output - y_train[i]
            theta = theta - lr * np.dot(x_train[i], error)

            predict_error = 0
            for j in range(num_test_samples):
                predict_output = sigmoid(np.dot(x_test[j], theta))
                predict_error -= y_test[j] * np.log(predict_output) + (1 - y_test[j]) * np.log(1 - predict_output)
            print(str(i) + '\t' + str(predict_error / num_test_samples))

            iters.append(i)
            test_errors.append(predict_error / num_test_samples)

    if method == 'nag':
        lr = 0.01
        miu = 0.9
        momentum = np.zeros(num_features + 1)

        for i in range(max_iterations):
            output = sigmoid(np.dot(x_train[i], theta - lr * miu * momentum))
            error = output - y_train[i]
            grad = np.dot(x_train[i], error)
            momentum = momentum * lr + grad
            theta = theta - lr * momentum

            predict_error = 0
            for j in range(num_test_samples):
                predict_output = sigmoid(np.dot(x_test[j], theta))
                predict_error -= y_test[j] * np.log(predict_output) + (1 - y_test[j]) * np.log(1 - predict_output)
            print(str(i) + '\t' + str(predict_error / num_test_samples))

            iters.append(i)
            test_errors.append(predict_error / num_test_samples)

    if method == 'rmsprop':
        lr = 0.1
        expectation = 1
        rho = 0.95
        delta = 10e-7

        for i in range(max_iterations):
            output = sigmoid(np.dot(x_train[i], theta))
            error = output - y_train[i]
            grad = np.dot(x_train[i], error)
            norm = grad * grad
            expectation = rho * expectation + (1 - rho) * norm
            theta = theta - lr * grad / (np.sqrt(expectation) + delta)

            predict_error = 0
            for j in range(num_test_samples):
                predict_output = sigmoid(np.dot(x_test[j], theta))
                predict_error -= y_test[j] * np.log(predict_output) + (1 - y_test[j]) * np.log(1 - predict_output)
            print(str(i) + '\t' + str(predict_error / num_test_samples))
            iters.append(i)
            test_errors.append(predict_error / num_test_samples)

    if method == 'adam':
        delta = 10e-8
        rho1 = 0.9
        rho2 = 0.999
        lr = 0.1
        s = 0
        r = 0

        for i in range(max_iterations):
            output = sigmoid(np.dot(x_train[i], theta))
            error = output - y_train[i]
            grad = np.dot(x_train[i], error)

            s = rho1 * s + (1 - rho1) * grad
            r = rho2 * r + (1 - rho2) * grad * grad
            s_hat = s / (1 - rho1)
            r_hat = r / (1 - rho2)
            delta_theta = (-lr * s_hat) / (np.sqrt(r_hat) + delta)
            theta = theta + delta_theta

            predict_error = 0
            for j in range(num_test_samples):
                predict_output = sigmoid(np.dot(x_test[j], theta))
                predict_error -= y_test[j] * np.log(predict_output) + (1 - y_test[j]) * np.log(1 - predict_output)
            print(str(i) + '\t' + str(predict_error / num_test_samples))
            iters.append(i)
            test_errors.append(predict_error / num_test_samples)

    if method == 'adadelta':
        r = 0
        e = 0
        miu = 0.9
        delta = 10e-7
        lr = 10

        for i in range(max_iterations):
            output = sigmoid(np.dot(x_train[i], theta))
            error = output - y_train[i]
            grad = np.dot(x_train[i], error)

            r = miu * r + (1 - miu) * grad * grad
            delta_theta = (-lr * grad * np.sqrt(e + delta)) / (np.sqrt(r + delta))
            theta = theta + delta_theta
            e = miu * e + (1 - miu) * e * e


            predict_error = 0
            for j in range(num_test_samples):
                predict_output = sigmoid(np.dot(x_test[j], theta))
                predict_error -= y_test[j] * np.log(predict_output) + (1 - y_test[j]) * np.log(1 - predict_output)
            print(str(i) + '\t' + str(predict_error / num_test_samples))
            iters.append(i)
            test_errors.append(predict_error / num_test_samples)

if __name__ == '__main__':
    x_train, y_train = ds.load_svmlight_file('./data/a9a')
    x_test, y_test = ds.load_svmlight_file('./data/a9a.t')

    num_samples, num_features = x_train.shape
    num_test_samples, num_test_features = x_test.shape

    x_train = x_train.toarray()
    temp = np.ones(shape=[32561, 1], dtype=np.float32)
    x_train = np.concatenate([x_train, temp], axis=1)
    x_test = x_test.toarray()
    temp = np.zeros(shape=[16281, 1], dtype=np.float32)
    temp1 = np.ones(shape=[16281, 1], dtype=np.float32)
    x_test = np.concatenate([x_test, temp, temp1], axis=1)

    for i in range(0, len(y_train)):
        if y_train[i] == -1:
            y_train[i] = 0
    for i in range(0, len(y_test)):
        if y_test[i] == -1:
            y_test[i] = 0


    methods = ['sgd', 'nag', 'rmsprop', 'adadelta', 'adam']
    for method in methods:
        iters = []
        test_errors = []
        train(x_train, y_train, x_test, y_test, method, iters, test_errors)
        plt.plot(iters, test_errors, label=method)

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
