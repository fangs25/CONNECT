import os
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


import argparse
parser = argparse.ArgumentParser(description='Model Running')
parser.add_argument('--dataset', default='RPE008', type=str,
                    help='datasets for training and testing')
parser.add_argument('--device', default='4', type = str, help = 'gpu index')
args = parser.parse_args()

# Dataset configuration
DATASET_NAME = args.dataset 

N_CELLS = 1000
N_GENES = 1500


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


def evaluate_single_dataset(adata, dataset_name, seeds=[0,1,2,3,4,5,6,7,8,9]):
# def evaluate_single_dataset(adata, dataset_name, seeds=[0,1]):

    rna = adata    

    methods = ['connect', 'totalvi', 'scpair', 'midas']
    available_methods = [m for m in methods if f'{m}_rna_predicted' in rna.uns]

    dataset_results = {}

    for seed in seeds:
        print(f"\n==============================")
        print(f" Dataset={dataset_name}, Seed={seed}")
        print("==============================")

        np.random.seed(seed)

        # ---- 抽样 ----
        cell_indices = np.random.choice(rna.shape[0], N_CELLS, replace=False)
        gene_indices = np.random.choice(rna.shape[1], N_GENES, replace=False)

        true_cell_data = rna.X[cell_indices, :]
        true_gene_data = rna.X[:, gene_indices]

        true_cell_corr = compute_correlation_matrix(true_cell_data)
        true_gene_corr = compute_correlation_matrix(true_gene_data.T)

        cell_norm = np.linalg.norm(true_cell_corr, 'fro')
        gene_norm = np.linalg.norm(true_gene_corr, 'fro')

        print(f"   Norm(Cell)={cell_norm:.4f}, Norm(Gene)={gene_norm:.4f}")

        seed_results = {}

        # ---- 遍历方法 ----
        for method in available_methods:
            print(f"\n   >> Method: {method}")

            try:
                pred = rna.uns[f'{method}_rna_predicted']

                # cellwise
                pred_cell_data = pred[cell_indices, :]
                pred_cell_corr = compute_correlation_matrix(pred_cell_data)

                cell_frob = compute_frobenius_distance(true_cell_corr, pred_cell_corr)
                cell_score = cell_frob / cell_norm if cell_norm > 0 else float("inf")

                # genewise
                pred_gene_data = pred[:, gene_indices]
                pred_gene_corr = compute_correlation_matrix(pred_gene_data.T)

                gene_frob = compute_frobenius_distance(true_gene_corr, pred_gene_corr)
                gene_score = gene_frob / gene_norm if gene_norm > 0 else float("inf")

                seed_results[method] = {
                    "cell_frobenius_norm": cell_score,
                    "gene_frobenius_norm": gene_score,
                }

                print(f"      ✔ cell={cell_score:.6f}, gene={gene_score:.6f}")

            except Exception as e:
                print(f"      ✘ failed: {e}")
                seed_results[method] = {"error": str(e)}

        dataset_results[f"seed_{seed}"] = seed_results

    # ---- 保存 JSON ----


    combined_file = f"./rna_prediction_results/cell_gene/{DATASET_NAME}_rna_atac_cell_gene_f_distances.json"
    os.makedirs(os.path.dirname(combined_file), exist_ok=True)

    with open(combined_file, 'w') as f:
        json.dump(dataset_results, f, indent=2)

    print(f"\nSaved → {combined_file}")
    return combined_file

rna_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/1117temp_midas_partialtrain/{DATASET_NAME}/rna_test.h5ad')
evaluate_single_dataset(rna_data, DATASET_NAME)