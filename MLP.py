__author__ = "Furkan Yakal"
__email__ = "fyakal16@ku.edu.tr"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Read the images and labels
images = pd.read_csv("hw03_images.csv", header=None)
labels = pd.read_csv("hw03_labels.csv", header=None)

# training set is formed from first 500 images
train_set_images = np.array(images[:500])  # (500, 784)
train_set_labels = np.array(labels[:500])  # (500, 1)

# test set is formed from remaining 500 images
test_set_images = np.array(images[500:])  # (500, 784)
test_set_labels = np.array(labels[500:])  # (500, 1)

# initialize weight parameters
W = np.array(pd.read_csv("initial_W.csv", header=None))  # (785, 20)
V = np.array(pd.read_csv("initial_V.csv", header=None))  # (21, 5)

# hyper_parameters
eta = 0.0005
epsilon = 1e-3
max_iteration = 500

# extracting known parameters
num_train_data = train_set_labels.shape[0]
num_test_data = test_set_labels.shape[0]
number_of_classes = len(np.unique(train_set_labels))


# adds one column to the given set
def add_one_column(matrix):
    return np.hstack((np.ones((matrix.shape[0], 1)), matrix))


# reformat the train labels into 0-1 matrix format
def one_hot_encode():
    y_correct = np.zeros((num_train_data, number_of_classes))
    for i in range(num_train_data):
        y_correct[i, train_set_labels[i] - 1] = 1
    return y_correct


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax function
def softmax(scores):
    scores -= np.max(scores, axis=1, keepdims=True)  # max score in each row is 0, makes numbers smaller for stability
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    return exp_scores / sum_exp_scores


def predict_class_scores(image_set, W=W, V=V):
    image_set = add_one_column(image_set)
    sigmoid_activation = add_one_column(sigmoid(image_set.dot(W)))
    softmax_activation = softmax(sigmoid_activation.dot(V))
    return image_set, softmax_activation, sigmoid_activation


# predicts the class of each image
def predict_classes(image_set, W=W, V=V):
    return np.argmax(predict_class_scores(image_set, W, V)[1], axis=1)


# using the error, updates the W and V matrices, plots the error graph
def error_and_weight_updates(image_set=train_set_images, W=W, V=V):
    y_correct = one_hot_encode()
    error_history = []

    for i in range(max_iteration):
        X, y_prediction, z = predict_class_scores(image_set, W, V)

        error = - np.sum(y_correct * np.log(y_prediction))
        error_history.append(error)

        gradient_V = np.matmul(z.T, y_prediction - y_correct)
        gradient_W = np.matmul(X.T, ((np.matmul(y_prediction - y_correct, V[1:].T)) * z[:, 1:] * (1 - z[:, 1:])))

        old_V = V
        old_W = W

        V -= eta * gradient_V
        W -= eta * gradient_W

        ''' 
        if np.sqrt(((V - old_V) ** 2).sum() + ((W - old_W) ** 2).sum()) < epsilon:
            break
        '''

    plot.plot(error_history)
    plot.xlabel('Iteration')
    plot.ylabel('Error')
    plot.show()


# generates the confusion matrix
def create_confusion_matrix(image_set, label_set):
    confusion_matrix = np.zeros((number_of_classes, number_of_classes))
    predictions = predict_classes(image_set)
    for i in range(len(label_set)):
        confusion_matrix[predictions[i], label_set[i] - 1] += 1
    return confusion_matrix


# prints the matrix in desired format
def print_confusion_matrix(data_set_type, matrix):
    print("\n----------------------{}----------------------\n".format(data_set_type))
    labeled_conf_matrix = pd.DataFrame(matrix.astype(int),
                                       index=['y_predicted T-shirt', 'y_predicted Trouser', 'y_predicted Dress',
                                              'y_predicted Sneaker', 'y_predicted Bag'],
                                       columns=['T-shirt', 'Trouser', 'Dress',
                                                'Sneaker', 'Bag'])
    print(labeled_conf_matrix)


# shape of the all data helpful while performing matrix operations
def shapes():
    print("Train_set_images: {}".format(train_set_images.shape))
    print("Train_set_labels: {}".format(train_set_labels.shape))
    print("Test_set_images: {}".format(test_set_images.shape))
    print("Test_set_labels: {}".format(test_set_labels.shape))
    print("W: {}".format(W.shape))
    print("V: {}".format(V.shape))


if __name__ == "__main__":
    # shapes()

    error_and_weight_updates()

    train_matrix = create_confusion_matrix(train_set_images, train_set_labels)
    print_confusion_matrix("TRAIN", train_matrix)

    test_matrix = create_confusion_matrix(test_set_images, test_set_labels)
    print_confusion_matrix("TEST", test_matrix)
