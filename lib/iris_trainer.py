import torch
from torch import nn
from torch.autograd import Variable

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import trange

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset():

    iris = load_iris()

    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    
    print(f"Shape of X (data): {X.shape}")
    print(f"Shape of y (target): {y.shape} {y.dtype}")
    print(f"Example of x and y pair: {X[0]} {y[0]}")

    # Scale data to have mean 0 and variance 1 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)

    print("Shape of training set X", X_train.shape)
    print("Shape of test set X", X_test.shape)

    return X, y, X_scaled, X_train, X_test, y_train, y_test

def train(X, y, model, loss_fn, optimizer):
    model.train()
    
    # convert numpy array to pytorch tensor
    X = Variable(torch.from_numpy(X)).float()
    y = Variable(torch.from_numpy(y)).long()
    X = X.to(device)
    y = y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.item()
    return loss

def predict(X, model):
    model.eval()
    with torch.no_grad():
        X = Variable(torch.from_numpy(X)).float()
        X = X.to(device)
        pred = model(X)
        pred = pred.argmax(1)
        pred = pred.cpu().detach().numpy()
    return pred

def test(X, y, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        X = Variable(torch.from_numpy(X)).float()
        y = Variable(torch.from_numpy(y)).long()
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= X.shape[0]
    return correct, test_loss

def train_model(model, epochs, X_train, X_test, y_train, y_test):
    torch.manual_seed(42)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_list     = np.zeros((epochs,))
    accuracy_list = np.zeros((epochs,))

    for epoch in trange(epochs):
        loss_list[epoch] = train(X_train, y_train, model, loss_fn, optimizer)
        correct, test_loss = test(X_test, y_test, model, loss_fn)
        accuracy_list[epoch] = correct
    
    print()
    print("Done. Accuracy:", accuracy_list[-1])
    return model

def show_plots(X, y, fixed_input = None, epsilon = None, title = ''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax1.plot(X_plot[:, 0], X_plot[:, 1], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.axis('equal')
    ax1.legend()

    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax2.plot(X_plot[:, 2], X_plot[:, 3], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax2.set_xlabel(feature_names[2])
    ax2.set_ylabel(feature_names[3])
    ax2.axis('equal')
    ax2.legend()

    if fixed_input_y is not None and epsilon is not None:
        #add rectangle to plot -> shows infinity norm 
        ax1.add_patch(Rectangle((fixed_input[0] - epsilon, fixed_input[1] - epsilon), 
                                2*epsilon, 2*epsilon, 
                                edgecolor='pink',
                                facecolor='none',      
                                lw=4))
        ax1.set_aspect("equal", adjustable="datalim")

        ax2.add_patch(Rectangle((fixed_input[2]-epsilon, fixed_input[3]-epsilon), 
                                2*epsilon, 2*epsilon, 
                                edgecolor='pink',
                                facecolor='none',      
                                lw=4))
        ax2.set_aspect("equal", adjustable="datalim")