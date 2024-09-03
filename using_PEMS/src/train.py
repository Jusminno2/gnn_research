import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from model import TrafficGNN
from scipy.sparse import load_npz


def load_data():
    x = np.load('data/processed/scaled_data.npy')
    adj = np.load('data/adj_matrices/adj_matrix.npy')
    # adj = load_npz('data/adj_matrices/adj_matrix.npz')

    edge_index, _ = from_scipy_sparse_matrix(adj)
    x = torch.tensor(x, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data


def train_model(data):
    model = TrafficGNN(num_features=data.num_features, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


if __name__ == "__main__":
    data = load_data()
    train_model(data)
