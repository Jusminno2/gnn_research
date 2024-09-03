import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


def split_data(data, train_size=0.7, val_size=0.15):
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=val_size / (1 - train_size), random_state=42)
    return train_data, val_data, test_data


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def plot_metrics(metrics, title='Training Metrics', ylabel='Value', xlabel='Epoch'):
    plt.figure(figsize=(10, 6))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_adj_matrix(data):
    num_nodes = data.shape[1]
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                adj_matrix[i, j] = np.exp(-np.linalg.norm(data[:, i] - data[:, j]))

    # Numpy arrayを疎行列形式（COO形式）に変換
    sparse_adj_matrix = coo_matrix(adj_matrix)
    return sparse_adj_matrix

def standardize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)
