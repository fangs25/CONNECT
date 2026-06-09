import os
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


import argparse
parser = argparse.ArgumentParser(description='Model Running')
parser.add_argument('--dataset', default='RPE008', type=str,
                    help='datasets for training and testing')
parser.add_argument('--device', default='3', type = str, help = 'gpu index')
args = parser.parse_args()

# Dataset configuration
DATASET_NAME = args.dataset 

N_CELLS = 1000
N_PROTEINS = 50


def compute_correlation_matrix(data):
    # Convert to dense if sparse
    if issparse(data):
        data = data.toarray()
        
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix

def compute_frobenius_distance(A, B):
    """Compute Frobenius distance"""
    return np.linalg.norm(A - B, 'fro')


def clr_normalize_seurat_style(adata, inplace=True):
    """Replicate R's CLR logic for ADT data"""

    if not inplace:
        adata = adata.copy()

    X = adata.X
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X.copy()

    for i in range(X_dense.shape[0]):
        x = X_dense[i, :].copy()
        nonzero_mask = x > 0
        nonzero_vals = x[nonzero_mask]
        
        # R logic ：Only calculate log1p for non-zero values, 
        #           with the denominator being the total characteristic number (including zero)
        sum_log_nonzero = np.log1p(nonzero_vals).sum()
        geom_mean_exp = np.exp(sum_log_nonzero / x.size)
        
        # log1p(x / geom_mean_exp)
        x_transformed = np.log1p(x / geom_mean_exp)
        
        X_dense[i, :] = x_transformed

    adata.X = X_dense.astype(np.float32)
    return adata

def apply_clr_to_prediction(predicted_data):
    """Apply CLR normalization to predicted ADT data"""
    # Convert to numpy if needed
    if hasattr(predicted_data, 'toarray'):
        predicted_data = predicted_data.toarray()
    elif hasattr(predicted_data, 'numpy'):
        predicted_data = predicted_data.numpy()

    # Handle numerical stability
    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-8
    predicted_data = np.maximum(predicted_data, epsilon)

    # Replace any infinite or NaN values
    predicted_data = np.where(np.isinf(predicted_data), epsilon, predicted_data)
    predicted_data = np.where(np.isnan(predicted_data), epsilon, predicted_data)

    # Create a temporary AnnData object for CLR normalization
    temp_adata = sc.AnnData(predicted_data)

    # Apply CLR normalization
    temp_adata = clr_normalize_seurat_style(temp_adata, inplace=True)

    # Get normalized data and ensure numerical stability
    normalized_data = temp_adata.X
    if hasattr(normalized_data, 'toarray'):
        normalized_data = normalized_data.toarray()

    # Replace any remaining infinite or NaN values
    normalized_data = np.where(np.isinf(normalized_data), 0.0, normalized_data)
    normalized_data = np.where(np.isnan(normalized_data), 0.0, normalized_data)

    return normalized_data



# def evaluate_single_dataset(adata, dataset_name, seeds=[0,1]):

def evaluate_single_dataset(adata, dataset_name, seeds=[0,1,2,3,4,5,6,7,8,9]):
    adt = adata.copy()
    adt_clr = clr_normalize_seurat_style(adt, inplace=True)    

    methods = ['connect', 'totalvi', 'scpair', 'midas']
    available_methods = [m for m in methods if f'{m}_adt_predicted' in adt_clr.uns]

    dataset_results = {}

    for seed in seeds:
        print(f"\n==============================")
        print(f" Dataset={dataset_name}, Seed={seed}")
        print("==============================")

        np.random.seed(seed)

        # ---- 抽样 ----
        cell_indices = np.random.choice(adt_clr.shape[0], N_CELLS, replace=False)
        pro_indices = np.random.choice(adt_clr.shape[1], N_PROTEINS, replace=False)

        true_cell_data = adt_clr.X[cell_indices, :]
        true_pro_data = adt_clr.X[:, pro_indices]

        true_cell_corr = compute_correlation_matrix(true_cell_data)
        true_pro_corr = compute_correlation_matrix(true_pro_data.T)

        cell_norm = np.linalg.norm(true_cell_corr, 'fro')
        pro_norm = np.linalg.norm(true_pro_corr, 'fro')

        print(f"   Norm(Cell)={cell_norm:.4f}, Norm(pro)={pro_norm:.4f}")

        seed_results = {}

        # ---- 遍历方法 ----
        for method in available_methods:
            print(f"\n   >> Method: {method}")

            try:
                if method == 'connect':
                    pred = adt_clr.uns[f'{method}_adt_predicted']  # Already CLR
                else:
                    adt_pred_original = adt_clr.uns[f'{method}_adt_predicted']
                    pred = apply_clr_to_prediction(adt_pred_original)
                # pred = adt_clr.uns[f'{method}_adt_predicted']

                # cellwise
                pred_cell_data = pred[cell_indices, :]
                pred_cell_corr = compute_correlation_matrix(pred_cell_data)

                cell_frob = compute_frobenius_distance(true_cell_corr, pred_cell_corr)
                cell_score = cell_frob / cell_norm if cell_norm > 0 else float("inf")

                # prowise
                pred_pro_data = pred[:, pro_indices]
                pred_pro_corr = compute_correlation_matrix(pred_pro_data.T)

                pro_frob = compute_frobenius_distance(true_pro_corr, pred_pro_corr)
                pro_score = pro_frob / pro_norm if pro_norm > 0 else float("inf")

                seed_results[method] = {
                    "cell_frobenius_norm": cell_score,
                    "pro_frobenius_norm": pro_score,
                }

                print(f"      ✔ cell={cell_score:.6f}, pro={pro_score:.6f}")

            except Exception as e:
                print(f"      ✘ failed: {e}")
                seed_results[method] = {"error": str(e)}

        dataset_results[f"seed_{seed}"] = seed_results

    # ---- 保存 JSON ----


    combined_file = f"./adt_prediction_results/cell_pro/{DATASET_NAME}_cell_pro_f_distances.json"
    os.makedirs(os.path.dirname(combined_file), exist_ok=True)

    with open(combined_file, 'w') as f:
        json.dump(dataset_results, f, indent=2)

    print(f"\nSaved → {combined_file}")
    return combined_file

adt_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/1117temp_midas_partialtrain/{DATASET_NAME}/adt_test.h5ad')
evaluate_single_dataset(adt_data, DATASET_NAME)