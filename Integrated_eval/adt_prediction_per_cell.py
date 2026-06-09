import numpy as np
import pandas as pd
import scanpy as sc
import os
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, wasserstein_distance
from scipy.sparse import issparse
from datetime import datetime

# Set scanpy settings
# sc.settings.verbosity = 1  # Reduced verbosity
# sc.settings.set_figure_params(dpi=80, facecolor='white')
import argparse
parser = argparse.ArgumentParser(description='Model Running')
parser.add_argument('--dataset', default='10xPBMC_raw', type=str,
                    help='datasets for training and testing')
parser.add_argument('--device', default='4', type = str, help = 'gpu index')
args = parser.parse_args()

# Dataset configuration
DATASET_NAME = args.dataset  # Change this to process different datasets
MODALITY = "RNA+ADT"


def load_adt_data(modality="RNA+ADT", dataset_name="issaacseq"):
    """Load RNA+ADT data and predictions for all methods"""
    
    print(f"Loading {modality} data from {dataset_name}...")
    
    # print(f"Loading RNA data from: {rna_file}")
    adt_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/1117temp_midas_partialtrain/{dataset_name}/adt_test.h5ad')
    adt = adt_data.copy()
    adt_clr = clr_normalize_seurat_style(adt, inplace=True)    

    print(f"RNA data shape: {adt_clr.shape}")
    
    # Extract prediction methods from adt_clr.uns
    available_methods = []
    predictions = {}
    
    for key in adt_clr.uns.keys():
        if key.endswith('_adt_predicted'):
            method = key.replace('_adt_predicted', '')
            available_methods.append(method)
            # pred_data = adt_data.uns[key]

            if method == 'connect':
                pred_data = adt_clr.uns[f'{method}_adt_predicted']  # Already CLR
            else:
                adt_pred_original = adt_clr.uns[f'{method}_adt_predicted']
                pred_data = apply_clr_to_prediction(adt_pred_original)
            
            # Convert to numpy array if needed
            if hasattr(pred_data, 'numpy'):
                pred_data = pred_data.numpy()
            
            predictions[method] = pred_data
            print(f"Found {method} predictions: shape {pred_data.shape}")
    
    print(f"Available prediction methods: {available_methods}")
    
    return adt_clr, predictions


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


def compute_per_cell_metrics(true_data, pred_data, method_name, batch_size=100):
    """
    Compute prediction quality metrics for each cell individually
    Based on the original py file implementation
    
    Parameters:
    - true_data: True RNA expression (cells x genes, sparse or dense)
    - pred_data: Predicted RNA expression (cells x genes, dense)
    - method_name: Name of the prediction method
    
    Returns:
    - DataFrame with per-cell metrics including method column
    """
    n_cells = true_data.shape[0]

    # Convert to dense format
    if issparse(true_data):
        true_data = true_data.toarray()
    if issparse(pred_data):
        pred_data = pred_data.toarray()

    metrics = {
        'cell_id': [],
        'method': [],
        'r2': [],
        'correlation': [],
        'mse': [],
        'mae': [],
        'cv_ratio': [],
        'wass_cell': np.zeros(n_cells)
    }

    print(f"Computing metrics for {n_cells} cells with {method_name} method...")

    # Process in batches for memory efficiency
    for i in range(0, n_cells, batch_size):
        end_idx = min(i + batch_size, n_cells)
        batch_true = true_data[i:end_idx]
        batch_pred = pred_data[i:end_idx]

        for j in range(batch_true.shape[0]):
            cell_idx = i + j
            true_vec = batch_true[j]
            pred_vec = batch_pred[j]

            # R² (from original py file)
            var_true = np.var(true_vec)
            if var_true > 1e-8:
                ss_res = np.sum((true_vec - pred_vec) ** 2)
                ss_tot = np.sum((true_vec - np.mean(true_vec)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                r2 = max(r2, -1.0)  # Limit to -1 minimum
            else:
                r2 = 0.0

            # Correlation
            try:
                if len(np.unique(true_vec)) > 1 and len(np.unique(pred_vec)) > 1:
                    corr, _ = pearsonr(true_vec, pred_vec)
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
            except:
                corr = 0.0

            # MSE
            mse = np.mean((true_vec - pred_vec) ** 2)

            # MAE
            mae = np.mean(np.abs(true_vec - pred_vec))

            # CV ratio for this specific cell
            true_mean = np.mean(true_vec)
            true_std = np.std(true_vec)
            pred_mean = np.mean(pred_vec)
            pred_std = np.std(pred_vec)
            
            true_cv = true_std / (true_mean + 1e-8) if true_mean > 0 else 0.0
            pred_cv = pred_std / (pred_mean + 1e-8) if pred_mean > 0 else 0.0
            cell_cv_ratio = pred_cv / (true_cv + 1e-8) if true_cv > 0 else 0.0


            t = true_vec.toarray().ravel() if issparse(true_vec) else true_vec
            p = pred_vec.toarray().ravel() if issparse(pred_vec) else pred_vec
            metrics['wass_cell'][cell_idx] = wasserstein_distance(t, p)


            # Store metrics
            metrics['cell_id'].append(cell_idx)
            metrics['method'].append(method_name)
            metrics['r2'].append(r2)
            metrics['correlation'].append(corr)
            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            metrics['cv_ratio'].append(cell_cv_ratio)

    return pd.DataFrame(metrics)

adt_data, predictions = load_adt_data(MODALITY, DATASET_NAME)


# Process all methods and combine into one DataFrame
all_metrics_dfs = []

for method_name, pred_data in predictions.items():
    print(f"\nProcessing {method_name} method...")
    
    # Check data shape compatibility
    print(f"True data shape: {adt_data.shape}")
    print(f"Pred data shape: {pred_data.shape}")

    if adt_data.shape != pred_data.shape:
        print(f"Warning: Shape mismatch, will use minimum dimensions")
        min_cells = min(adt_data.shape[0], pred_data.shape[0])
        min_genes = min(adt_data.shape[1], pred_data.shape[1])
        
        # Slice both datasets to matching dimensions
        true_subset = adt_data.X[:min_cells, :min_genes]
        pred_subset = pred_data[:min_cells, :min_genes]
    else:
        true_subset = adt_data.X
        pred_subset = pred_data
    
    # Compute metrics for this method
    method_metrics = compute_per_cell_metrics(true_subset, pred_subset, method_name)
    all_metrics_dfs.append(method_metrics)
    
    print(f"{method_name} metrics computed: {len(method_metrics)} cells")
    print(f"{method_name} metrics summary:")
    print(method_metrics[['r2', 'correlation', 'mse', 'mae', 'cv_ratio', 'wass_cell']].describe())

# Combine all method metrics into one DataFrame
if all_metrics_dfs:
    combined_metrics = pd.concat(all_metrics_dfs, ignore_index=True)
    
    print(f"\nCombined metrics shape: {combined_metrics.shape}")
    print(f"Methods included: {combined_metrics['method'].unique()}")
    print(f"Cells per method:")
    print(combined_metrics['method'].value_counts())
    
    # Save combined metrics to CSV with dataset name
    output_file = f"./adt_prediction_results/per_cell/{DATASET_NAME}_per_cell_metrics.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_metrics.to_csv(output_file, index=False)
    
    print(f"\nCombined metrics saved to: {output_file}")
    print(f"File contains {len(combined_metrics)} rows with columns: {list(combined_metrics.columns)}")
    
    # Display first few rows
    print(f"\nFirst few rows of combined metrics:")
    print(combined_metrics.head())
    
else:
    print("No metrics computed - no prediction methods found")
