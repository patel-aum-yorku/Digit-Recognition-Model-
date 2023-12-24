import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import random
from PIL import Image
from scipy import ndimage

def load_data(train_set_percentage):
    """
    Arguments:
    train_set_percentage -- python float indicating the percentage of the data to be used as training set
    
    Returns:
    train_set_x -- numpy array which represent the training set (digit images)
    train_set_y -- numpy array which represent the label of the training set
    test_set_x -- numpy array which represent the testing set (digit images)
    test_set_y -- numpy array which represent the label of the testing set
    """
    dataset = h5py.File('dataset.h5', "r")
    
    total_num = np.shape(np.array(dataset["digit_images"][:]))[1]
    train_num = int(total_num * train_set_percentage)
    test_num = total_num - train_num
    
    digits = dataset["digit_images"][:]
    labels = dataset["digit_labels"][:]
    
    train_set_x = np.zeros((np.shape(digits)[0], train_num))
    train_set_y = np.zeros((1, train_num))
    
    test_set_x = np.zeros((np.shape(digits)[0], test_num))
    test_set_y = np.zeros((1, test_num))
    
    training_samples = []
    random.seed(0)
    
    for i in range(train_num):
        found = False
        while not found:
            sample_index = random.randint(0, total_num - 1)
            if sample_index not in training_samples:
                training_samples.append(sample_index)
                found = True
                train_set_x[:, i] = digits[:, sample_index]
                train_set_y[:, i] = labels[:, sample_index]
    
    test_index = 0
    for i in range(total_num):
        if i not in training_samples:
            test_set_x[:, test_index] = digits[:, i]
            test_set_y[:, test_index] = labels[:, i]
            test_index += 1
    
    dataset.close()
    
    return train_set_x, train_set_y, test_set_x, test_set_y

def reshape_Y(Y):
    """
    Arguments:
    Y -- numpy array which represents the labels,
         example, [0, 1, 2, 3, ...]
    
    Returns:
    Y_output -- numpy array which represent the labels in a different way (corresponding to the output layer)
         example corresponding to above:
         [[1, 0, 0, 0, ...],
          [0, 1, 0, 0, ...],
          [0, 0, 1, 0, ...],
          [0, 0, 0, 0, ...],
          [0, 0, 0, 1, ...],
          [0, 0, 0, 0, ...],
          [0, 0, 0, 0, ...],
          [0, 0, 0, 0, ...],
          [0, 0, 0, 0, ...],
          [0, 0, 0, 0, ...],
    """    
    num_samples = int(np.shape(Y)[1])
    Y_output = np.zeros((10, num_samples))
    
    for i in range(np.shape(Y)[1]):
        Y_output[int(np.squeeze(Y[:, i])), i] = 1

    return Y_output

def display_digit_image(image_set, label_set, index):
    """
    Arguments:
    image_set -- numpy array which represents the image
    label_set -- numpy array which represents the label
    index -- the index of the image to be shown
    figure_index -- the index of the figure to show upon (every time when calling plt.figure function, a new figure_index needs to be assigned to get a new figure rather than draw on the original figure)
    """
    image = image_set[:, index].reshape(20, 20).T
    plt.imshow(image)
    plt.title("digit is " + str(int(label_set[0, index])))
    plt.xlim(0, 19)
    plt.ylim(19, 0)
    plt.xticks(np.arange(0, 20, 2))
    plt.yticks(np.arange(0, 20, 2))

def display_cost(costs, learning_rate, figure_index):
    """
    display the cost in a figure
    
    Arguments:
    costs -- a list of floats with the costs of each iteration
    learning_rate -- learning rate to be displayed on the figure
    figure_index -- figure_index -- the index of the figure to show upon (every time when calling plt.figure function, a new figure_index needs to be assigned to get a new figure rather than draw on the original figure)
    """    
    plt.figure(figure_index)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()   

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (0, 1, 2, 3 ...)
    """
    A2, cache = feedforward(X, parameters)
    predictions = np.argmax(A2, axis = 0)
    np.reshape(predictions, (1, np.shape(predictions)[0]))
    
    return predictions

def compute_accuracy(predictions, results):
    """
    Arguments:
    predictions -- output of predict
    results -- true labels
    
    Returns
    accuracy -- how accurate this model is to predict the digits
    """    
    comparison = predictions == results
    return np.sum(comparison) / np.shape(results)[1]
