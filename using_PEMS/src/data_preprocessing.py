import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from scipy.sparse import save_npz


def load_raw_data():
    data = pd.read_csv('data/raw/PEMSd7.csv')
    return data


def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)

    # グラフの隣接行列を計算
    adj_matrix = calculate_adj_matrix(scaled_data)

    # データを保存
    np.save('data/processed/scaled_data.npy', scaled_data)
    save_npz('data/adj_matrices/adj_matrix.npz', adj_matrix)


def calculate_adj_matrix(data):
    # サンプルとして、距離に基づく単純な隣接行列を作成
    num_nodes = data.shape[1]
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                adj_matrix[i, j] = np.exp(-np.linalg.norm(data[:, i] - data[:, j]))

    return adj_matrix


if __name__ == "__main__":
    raw_data = load_raw_data()
    preprocess_data(raw_data)
