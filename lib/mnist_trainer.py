from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_gtsrb_dataset(batch_size=64):
    # Download training and test data
    # https://pytorch.org/vision/main/generated/torchvision.datasets.GTSRB.html
    training_data = datasets.GTSRB(
        root="data",
        split="train",
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.GTSRB(
        root="data",
        split="test",
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader


def load_mnist_dataset(batch_size=64):
    # Download training and test data
    # https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader


class CensusIncomeDataset(Dataset):
    def prepare_df(self, filename):
        columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                   "income"]
        df = pd.read_csv(filename, names=columns, sep=", ", engine='python')

        # drop missing values
        df = df.replace({'?': np.nan}).dropna()

        # Encode Data
        df.workclass.replace(('Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                              'Without-pay', 'Never-worked'), (1, 2, 3, 4, 5, 6, 7, 8), inplace=True)
        df.education.replace(('Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                              '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th',
                              'Preschool'), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), inplace=True)
        df["marital-status"].replace(('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                                      'Married-spouse-absent', 'Married-AF-spouse'), (1, 2, 3, 4, 5, 6, 7),
                                     inplace=True)
        df.occupation.replace(('Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                               'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                               'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                               'Armed-Forces'), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), inplace=True)
        df.relationship.replace(('Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'),
                                (1, 2, 3, 4, 5, 6), inplace=True)
        df.race.replace(('White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'), (1, 2, 3, 4, 5),
                        inplace=True)
        df.sex.replace(('Female', 'Male'), (1, 2), inplace=True)
        df["native-country"].replace(('United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                                      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                                      'Cuba',
                                      'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
                                      'Mexico',
                                      'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                                      'Taiwan',
                                      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
                                      'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                                      'Holand-Netherlands'), (
                                         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                         23,
                                         24,
                                         25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41),
                                     inplace=True)
        df.income.replace(('<=50K', '>50K'), (0, 1), inplace=True)
        df = df.astype("int64")
        return df

    def __init__(self, filename):
        df = self.prepare_df(filename)
        x = df.iloc[:, 0:14].values
        y = df.iloc[:, 14].values

        self.x_train = torch.tensor(x, dtype=torch.int64)
        self.y_train = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


def load_census_income_dataset(batch_size=64):
    # https://archive.ics.uci.edu/ml/datasets/census+income
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    r = requests.get(url)
    Path("data/census/").mkdir(parents=True, exist_ok=True)
    filename = "data/census/adult.data"
    with open(filename, 'w') as file:
        file.write(r.text)

    data = CensusIncomeDataset(filename)
    # Create data loaders.
    train_dataloader = DataLoader(data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss


def train_model(model, epochs, train_dataloader, test_dataloader):
    torch.manual_seed(42)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        correct, test_loss = test(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    print("Done!")
    return model
