import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification

#initialize parameters
learning_rate_1 = 0.01
#Create dataset
X, y = make_classification(n_samples=500, n_features=3, n_informative=3,
                           n_redundant=0, n_clusters_per_class=1,
                           flip_y=0.1,  # adds a small amount of noise
                           class_sep=1.0,  # classes are separable but not too easily
                           random_state=40)
#relabel the Y targets to 1/-1
y = np.where(y == 0, -1, y)
#Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.25,shuffle=True)

#3Implement soft margin svm. Compute the loss function
def hinge_loss (x_1, y_1,w,b,c):
    margin = 1 - y_1 * (np.dot(w, x_1) + b)
    hinge = max(0, margin)
    return 0.5 * np.dot(w, w) + c * hinge

def gradient_w (w,b,x_1,y_1,c) :
    if y_1 * (np.dot(w, x_1) + b) < 1:
        return w - c * y_1 * x_1  # Misclassified or within margin
    else:
        return w  # Correctly classified, no update


def gradient_b(w,b,x_1,y_1): 
    if y_1 * (np.dot(w, x_1) + b) < 1:
        return -c * y_1  # Misclassified or within margin
    else:
        return 0  # Correctly classified, no update

#3.Implement soft margin SVM
def svm_mini_batch_gradient(x_1, y_1, c, lr, epochs, batch_s):
    hinge_list=[]
    # Initialize weights and bias : Shuffle the dataset before creating mini-batches
     # Initialize gradients for weights and bias
    grad_w = np.zeros(X.shape[1])
    grad_b = 0
    n = X.shape[0]  # Number of samples
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        x_shuffled = x_1[indices]
        y_shuffled = y_1[indices]
        # Perform gradient descent
        inter_ite =0
        dw, db = 0, 0
        for x_2, y_2 in zip(x_1, y_1):
            # Check if the point violates the margin (i.e., hinge loss is non-zero)
            dw += gradient_w(w, b, x_2, y_2)
            db += gradient_b(w, b, x_2, y_2)
            inter_ite = inter_ite + 1
        #check if the batch is complete to add the necessary values
            if inter_ite % mini_batch_size == 0:
                w = w - lr * dw
                b = b - lr * db
                #calculate the loss function after each epoch
                hinge_list.append(hinge_loss (x_1, y_1,w,b,c))
                # reset gradients 
                dw, db = 0, 0
    return w, b, hinge_list

# Example usage:
# X is the feature matrix, y is the label vector (+1/-1)
# C is the regularization parameter, lr is the learning rate, epochs is the number of iterations


#4.Use mini batch gradient descent to minimize the loss function on the next page (shuffle the data first)
#5.Return the optimal weights by minimizing the loss function

#6.Perform some predictions on the test data
C = 1.0  # Regularization parameter
learning_rate_2 = 0.01
epochs_trial = 100
batch_size = 32

Weight, Bias, losses = svm_mini_batch_gradient(X_train, y_train, C, learning_rate_2, epochs_trial, batch_size)

def predict(x_2, w_1, b_1):
    return np.sign(np.dot(x_2, w_1) + b_1)

y_predict = predict(X_test, Weight, Bias)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy * 100:.2f}%")


#8.Visualize the training data and decision boundary in 3D



#9.Visualize the loss function over time during training