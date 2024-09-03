# GNN Traffic Prediction

This project aims to predict traffic flow using Graph Neural Networks (GNNs) with the PEMS dataset.

## Project Structure

- `data/`
  - `raw/`: Raw PEMS dataset
  - `processed/`: Preprocessed data
  - `adj_matrices/`: Adjacency matrices for the graph

- `src/`
  - `data_preprocessing.py`: Script for data preprocessing
  - `model.py`: Defines the GNN model
  - `train.py`: Training script for the model
  - `utils.py`: Utility functions

- `notebooks/`
  - `exploratory_data_analysis.ipynb`: Notebook for exploratory data analysis

## Setup Instructions

1. Create a conda environment:
    ```bash
    conda create -n gnn_traffic python=3.10
    conda activate gnn_traffic
    ```

2. Install the required packages:
    ```bash
    conda install -c conda-forge numpy pandas scikit-learn matplotlib networkx
    pip install torch torch-geometric
    ```

3. Download the PEMS dataset and place it in the `data/raw/` directory.

4. Run data preprocessing:
    ```bash
    python src/data_preprocessing.py
    ```

5. Train the model:
    ```bash
    python src/train.py
    ```

## Notes

- Adjust the hyperparameters in `train.py` as needed.
- Use the `notebooks/exploratory_data_analysis.ipynb` to visualize and analyze the data.
