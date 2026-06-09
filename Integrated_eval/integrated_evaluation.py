#!/usr/bin/env python3
"""
Integrated Evaluation for Single-cell Multi-omic Data
整合单细胞配对数据的五个评估方面为一个统一文件
支持指定method和结果追加保存
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import time
import scanpy as sc
from scipy.spatial.distance import cdist
import hnswlib
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.sparse import issparse
from datetime import datetime
import argparse

try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps


# ==================== 复用的函数 ====================

def FOSFTTM_batch(query, reference, batch_size=500, output_full=False):
    """Compute FOSFTTM scores as originally defined"""
    n = reference.shape[0]
    m = query.shape[0]
    fraction = []
    for batch_start in range(0, m, batch_size):
        batch_end = min(batch_start + batch_size, m)
        query_batch = query[batch_start:batch_end]
        distances = cdist(query_batch, reference, metric='sqeuclidean')
        for i in range(batch_start, batch_end):
            fraction.append(np.sum(distances[i - batch_start, i] < distances[i - batch_start, :]) / (n - 1))
    if output_full:
        return np.array(fraction)
    avg_fraction = np.sum(fraction) / m
    return avg_fraction


def top_k_matching_accuracy_hnsw(mapped_embeddings, target_embeddings, k=1, ef=200, M=16):
    """Compute top-k matching accuracy using HNSW"""
    mapped_embeddings = mapped_embeddings.astype(np.float32)
    target_embeddings = target_embeddings.astype(np.float32)
    dim = target_embeddings.shape[1]
    num_elements = target_embeddings.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)
    p.add_items(target_embeddings, np.arange(num_elements))
    p.set_ef(ef)
    indices, distances = p.knn_query(mapped_embeddings, k=k)
    correct = np.sum(np.arange(mapped_embeddings.shape[0])[:, None] == indices)
    return correct / mapped_embeddings.shape[0]


def compute_correlation_matrix(data):
    """Compute correlation matrix for given data"""
    if issparse(data):
        data = data.toarray()
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

        sum_log_nonzero = np.log1p(nonzero_vals).sum()
        geom_mean_exp = np.exp(sum_log_nonzero / x.size)

        x_transformed = np.log1p(x / geom_mean_exp)
        X_dense[i, :] = x_transformed

    adata.X = X_dense.astype(np.float32)
    return adata


def apply_clr_to_prediction(predicted_data):
    """Apply CLR normalization to predicted ADT data"""
    if hasattr(predicted_data, 'toarray'):
        predicted_data = predicted_data.toarray()
    elif hasattr(predicted_data, 'numpy'):
        predicted_data = predicted_data.numpy()

    epsilon = 1e-8
    predicted_data = np.maximum(predicted_data, epsilon)
    predicted_data = np.where(np.isinf(predicted_data), epsilon, predicted_data)
    predicted_data = np.where(np.isnan(predicted_data), epsilon, predicted_data)

    temp_adata = sc.AnnData(predicted_data)
    temp_adata = clr_normalize_seurat_style(temp_adata, inplace=True)

    normalized_data = temp_adata.X
    if hasattr(normalized_data, 'toarray'):
        normalized_data = normalized_data.toarray()

    normalized_data = np.where(np.isinf(normalized_data), 0.0, normalized_data)
    normalized_data = np.where(np.isnan(normalized_data), 0.0, normalized_data)

    return normalized_data


# ==================== 主要评估类 ====================

class IntegratedMultiModalEvaluator:
    """整合的多模态单细胞数据评估器"""

    def __init__(self, output_file="integrated_evaluation_results.csv"):
        self.output_file = output_file
        self.columns = [
            "dataset", "method", "timestamp",
            # Mapping指标
            "foscttm_rna_mean", "foscttm_rna_std", "foscttm_adt_mean", "foscttm_adt_std",
            "rna_area_normalized", "peak_area_normalized",
            # RNA预测指标 - 细胞级别
            "rna_per_cell_r2_mean", "rna_per_cell_r2_std",
            "rna_per_cell_correlation_mean", "rna_per_cell_correlation_std",
            "rna_per_cell_mse_mean", "rna_per_cell_mse_std",
            "rna_per_cell_mae_mean", "rna_per_cell_mae_std",
            "rna_per_cell_cv_ratio_mean", "rna_per_cell_cv_ratio_std",
            "rna_per_cell_wass_mean", "rna_per_cell_wass_std",
            # RNA预测指标 - 细胞/基因Frobenius距离
            "rna_cell_frobenius_mean", "rna_cell_frobenius_std",
            "rna_gene_frobenius_mean", "rna_gene_frobenius_std",
            # ADT预测指标 - 细胞级别
            "adt_per_cell_r2_mean", "adt_per_cell_r2_std",
            "adt_per_cell_correlation_mean", "adt_per_cell_correlation_std",
            "adt_per_cell_mse_mean", "adt_per_cell_mse_std",
            "adt_per_cell_mae_mean", "adt_per_cell_mae_std",
            "adt_per_cell_cv_ratio_mean", "adt_per_cell_cv_ratio_std",
            "adt_per_cell_wass_mean", "adt_per_cell_wass_std",
            # ADT预测指标 - 细胞/蛋白Frobenius距离
            "adt_cell_frobenius_mean", "adt_cell_frobenius_std",
            "adt_pro_frobenius_mean", "adt_pro_frobenius_std"
        ]

        print("Integrated Multi-modal Evaluator 初始化完成")
        print(f"输出文件: {self.output_file}")

    def load_data(self, dataset_name: str) -> Dict[str, Any]:
        """加载RNA和ADT数据"""
        print(f"加载数据集: {dataset_name}")

        try:
            # 加载RNA数据
            rna_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/1205midas_map/{dataset_name}/rna_test.h5ad')
            print(f"✓ RNA数据加载成功: {rna_data.shape}")

            # 加载ADT数据
            adt_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/1117temp_midas_partialtrain/{dataset_name}/adt_test.h5ad')
            print(f"✓ ADT数据加载成功: {adt_data.shape}")

            # 获取可用方法
            available_methods = set()

            # RNA相关方法
            for key in rna_data.uns.keys():
                if any(key.endswith(suffix) for suffix in ['_rna_predicted', '_rna_emb', '_rna_mapped']):
                    method = key.split('_rna')[0]
                    available_methods.add(method)

            # ADT相关方法
            for key in adt_data.uns.keys():
                if any(key.endswith(suffix) for suffix in ['_adt_predicted', '_adt_emb', '_adt_mapped']):
                    method = key.split('_adt')[0]
                    available_methods.add(method)

            available_methods = list(available_methods)
            print(f"✓ 可用方法: {available_methods}")

            return {
                'rna_data': rna_data,
                'adt_data': adt_data,
                'available_methods': available_methods,
                'dataset_name': dataset_name
            }

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None

    def evaluate_mapping_metrics(self, rna_data, adt_data, method: str) -> Dict[str, float]:
        """评估mapping指标"""
        print(f"  评估mapping指标...")

        try:
            # 获取映射结果
            rna_emb = rna_data.uns.get(f'{method}_rna_emb', None)
            adt_emb = adt_data.uns.get(f'{method}_adt_emb', None)
            rna_mapped = rna_data.uns.get(f'{method}_rna_mapped', None)
            adt_mapped = adt_data.uns.get(f'{method}_adt_mapped', None)

            if rna_mapped is None or adt_mapped is None:
                return self._empty_mapping_metrics()

            # 转换为numpy数组
            if hasattr(rna_mapped, 'numpy'):
                rna_mapped = rna_mapped.numpy()
            if hasattr(adt_mapped, 'numpy'):
                adt_mapped = adt_mapped.numpy()
            if hasattr(rna_emb, 'numpy'):
                rna_emb = rna_emb.numpy()
            if hasattr(adt_emb, 'numpy'):
                adt_emb = adt_emb.numpy()

            # 计算FOSCTTM
            foscttm_rna_full = FOSFTTM_batch(rna_mapped, rna_emb, output_full=True)
            foscttm_adt_full = FOSFTTM_batch(adt_mapped, adt_emb, output_full=True)

            # 计算Top-K Accuracy
            n_cells = rna_emb.shape[0]
            rate_list = np.arange(0.5, 10.1, 0.5)
            rate_based_k = [int(n_cells * rate * 0.01) for rate in rate_list]
            fixed_k = [1, 2, 3, 4, 5, 8, 10]
            all_k_values = sorted(set(rate_based_k + fixed_k))
            all_k_values = [k for k in all_k_values if k < n_cells and k > 0]

            top_k_results = []
            for k in all_k_values:
                k_rna = top_k_matching_accuracy_hnsw(rna_mapped, rna_emb, k)
                k_peak = top_k_matching_accuracy_hnsw(adt_mapped, adt_emb, k)
                top_k_results.append({'k_value': k, 'rna_accuracy': k_rna, 'peak_accuracy': k_peak})

            # 计算面积
            current_results_df = pd.DataFrame(top_k_results).sort_values(by='k_value')
            rna_area = simps(current_results_df['rna_accuracy'], current_results_df['k_value'])
            peak_area = simps(current_results_df['peak_accuracy'], current_results_df['k_value'])

            max_k = max(all_k_values)
            rna_area_normalized = rna_area / max_k
            peak_area_normalized = peak_area / max_k

            return {
                'foscttm_rna_mean': np.mean(foscttm_rna_full),
                'foscttm_rna_std': np.std(foscttm_rna_full),
                'foscttm_adt_mean': np.mean(foscttm_adt_full),
                'foscttm_adt_std': np.std(foscttm_adt_full),
                'rna_area_normalized': rna_area_normalized,
                'peak_area_normalized': peak_area_normalized
            }

        except Exception as e:
            print(f"    ❌ mapping指标计算失败: {e}")
            return self._empty_mapping_metrics()

    def evaluate_rna_prediction_per_cell(self, rna_data, method: str) -> Dict[str, float]:
        """评估RNA预测的细胞级别指标"""
        print(f"  评估RNA预测细胞级别指标...")

        try:
            pred_data = rna_data.uns.get(f'{method}_rna_predicted', None)
            if pred_data is None:
                return self._empty_per_cell_metrics("rna")

            if hasattr(pred_data, 'numpy'):
                pred_data = pred_data.numpy()

            true_data = rna_data.X
            if issparse(true_data):
                true_data = true_data.toarray()

            # 检查形状兼容性
            if true_data.shape != pred_data.shape:
                min_cells = min(true_data.shape[0], pred_data.shape[0])
                min_genes = min(true_data.shape[1], pred_data.shape[1])
                true_data = true_data[:min_cells, :min_genes]
                pred_data = pred_data[:min_cells, :min_genes]

            # 计算指标
            r2_scores, correlations, mses, maes, cv_ratios, wass_dists = [], [], [], [], [], []

            for i in range(true_data.shape[0]):
                true_vec = true_data[i]
                pred_vec = pred_data[i]

                # R²
                var_true = np.var(true_vec)
                if var_true > 1e-8:
                    ss_res = np.sum((true_vec - pred_vec) ** 2)
                    ss_tot = np.sum((true_vec - np.mean(true_vec)) ** 2)
                    r2 = max(1 - (ss_res / ss_tot), -1.0)
                else:
                    r2 = 0.0
                r2_scores.append(r2)

                # Correlation
                try:
                    if len(np.unique(true_vec)) > 1 and len(np.unique(pred_vec)) > 1:
                        corr, _ = pearsonr(true_vec, pred_vec)
                        corr = 0.0 if np.isnan(corr) else corr
                    else:
                        corr = 0.0
                except:
                    corr = 0.0
                correlations.append(corr)

                # MSE和MAE
                mses.append(np.mean((true_vec - pred_vec) ** 2))
                maes.append(np.mean(np.abs(true_vec - pred_vec)))

                # CV ratio
                true_mean, true_std = np.mean(true_vec), np.std(true_vec)
                pred_mean, pred_std = np.mean(pred_vec), np.std(pred_vec)
                true_cv = true_std / (true_mean + 1e-8) if true_mean > 0 else 0.0
                pred_cv = pred_std / (pred_mean + 1e-8) if pred_mean > 0 else 0.0
                cv_ratios.append(pred_cv / (true_cv + 1e-8) if true_cv > 0 else 0.0)

                # Wasserstein distance
                wass_dists.append(wasserstein_distance(true_vec, pred_vec))

            return {
                'rna_per_cell_r2_mean': np.mean(r2_scores),
                'rna_per_cell_r2_std': np.std(r2_scores),
                'rna_per_cell_correlation_mean': np.mean(correlations),
                'rna_per_cell_correlation_std': np.std(correlations),
                'rna_per_cell_mse_mean': np.mean(mses),
                'rna_per_cell_mse_std': np.std(mses),
                'rna_per_cell_mae_mean': np.mean(maes),
                'rna_per_cell_mae_std': np.std(maes),
                'rna_per_cell_cv_ratio_mean': np.mean(cv_ratios),
                'rna_per_cell_cv_ratio_std': np.std(cv_ratios),
                'rna_per_cell_wass_mean': np.mean(wass_dists),
                'rna_per_cell_wass_std': np.std(wass_dists)
            }

        except Exception as e:
            print(f"    ❌ RNA预测细胞级别指标计算失败: {e}")
            return self._empty_per_cell_metrics("rna")

    def evaluate_adt_prediction_per_cell(self, adt_data, method: str) -> Dict[str, float]:
        """评估ADT预测的细胞级别指标"""
        print(f"  评估ADT预测细胞级别指标...")

        try:
            pred_original = adt_data.uns.get(f'{method}_adt_predicted', None)
            if pred_original is None:
                return self._empty_per_cell_metrics("adt")

            # CLR标准化
            if method == 'connect':
                pred_data = pred_original  # 已经是CLR
            else:
                pred_data = apply_clr_to_prediction(pred_original)

            true_data = adt_data.X
            if issparse(true_data):
                true_data = true_data.toarray()

            # 检查形状兼容性
            if true_data.shape != pred_data.shape:
                min_cells = min(true_data.shape[0], pred_data.shape[0])
                min_proteins = min(true_data.shape[1], pred_data.shape[1])
                true_data = true_data[:min_cells, :min_proteins]
                pred_data = pred_data[:min_cells, :min_proteins]

            # 计算指标（同RNA）
            r2_scores, correlations, mses, maes, cv_ratios, wass_dists = [], [], [], [], [], []

            for i in range(true_data.shape[0]):
                true_vec = true_data[i]
                pred_vec = pred_data[i]

                # R²
                var_true = np.var(true_vec)
                if var_true > 1e-8:
                    ss_res = np.sum((true_vec - pred_vec) ** 2)
                    ss_tot = np.sum((true_vec - np.mean(true_vec)) ** 2)
                    r2 = max(1 - (ss_res / ss_tot), -1.0)
                else:
                    r2 = 0.0
                r2_scores.append(r2)

                # Correlation
                try:
                    if len(np.unique(true_vec)) > 1 and len(np.unique(pred_vec)) > 1:
                        corr, _ = pearsonr(true_vec, pred_vec)
                        corr = 0.0 if np.isnan(corr) else corr
                    else:
                        corr = 0.0
                except:
                    corr = 0.0
                correlations.append(corr)

                # MSE和MAE
                mses.append(np.mean((true_vec - pred_vec) ** 2))
                maes.append(np.mean(np.abs(true_vec - pred_vec)))

                # CV ratio
                true_mean, true_std = np.mean(true_vec), np.std(true_vec)
                pred_mean, pred_std = np.mean(pred_vec), np.std(pred_vec)
                true_cv = true_std / (true_mean + 1e-8) if true_mean > 0 else 0.0
                pred_cv = pred_std / (pred_mean + 1e-8) if pred_mean > 0 else 0.0
                cv_ratios.append(pred_cv / (true_cv + 1e-8) if true_cv > 0 else 0.0)

                # Wasserstein distance
                wass_dists.append(wasserstein_distance(true_vec, pred_vec))

            return {
                'adt_per_cell_r2_mean': np.mean(r2_scores),
                'adt_per_cell_r2_std': np.std(r2_scores),
                'adt_per_cell_correlation_mean': np.mean(correlations),
                'adt_per_cell_correlation_std': np.std(correlations),
                'adt_per_cell_mse_mean': np.mean(mses),
                'adt_per_cell_mse_std': np.std(mses),
                'adt_per_cell_mae_mean': np.mean(maes),
                'adt_per_cell_mae_std': np.std(maes),
                'adt_per_cell_cv_ratio_mean': np.mean(cv_ratios),
                'adt_per_cell_cv_ratio_std': np.std(cv_ratios),
                'adt_per_cell_wass_mean': np.mean(wass_dists),
                'adt_per_cell_wass_std': np.std(wass_dists)
            }

        except Exception as e:
            print(f"    ❌ ADT预测细胞级别指标计算失败: {e}")
            return self._empty_per_cell_metrics("adt")

    def evaluate_rna_frobenius_distances(self, rna_data, method: str, n_seeds=10,
                                        n_cells=1000, n_genes=1500) -> Dict[str, float]:
        """评估RNA预测的细胞和基因Frobenius距离"""
        print(f"  评估RNA预测Frobenius距离...")

        try:
            pred_data = rna_data.uns.get(f'{method}_rna_predicted', None)
            if pred_data is None:
                return self._empty_frobenius_metrics("rna")

            cell_scores, gene_scores = [], []

            for seed in range(n_seeds):
                np.random.seed(seed)

                # 抽样
                cell_indices = np.random.choice(rna_data.shape[0], n_cells, replace=False)
                gene_indices = np.random.choice(rna_data.shape[1], n_genes, replace=False)

                # 真实数据
                true_cell_data = rna_data.X[cell_indices, :]
                true_gene_data = rna_data.X[:, gene_indices]

                if issparse(true_cell_data):
                    true_cell_data = true_cell_data.toarray()
                if issparse(true_gene_data):
                    true_gene_data = true_gene_data.toarray()

                true_cell_corr = compute_correlation_matrix(true_cell_data)
                true_gene_corr = compute_correlation_matrix(true_gene_data.T)

                cell_norm = np.linalg.norm(true_cell_corr, 'fro')
                gene_norm = np.linalg.norm(true_gene_corr, 'fro')

                # 预测数据
                pred_cell_data = pred_data[cell_indices, :]
                pred_gene_data = pred_data[:, gene_indices]

                pred_cell_corr = compute_correlation_matrix(pred_cell_data)
                pred_gene_corr = compute_correlation_matrix(pred_gene_data.T)

                # 计算Frobenius距离
                cell_frob = compute_frobenius_distance(true_cell_corr, pred_cell_corr)
                gene_frob = compute_frobenius_distance(true_gene_corr, pred_gene_corr)

                cell_score = cell_frob / cell_norm if cell_norm > 0 else float("inf")
                gene_score = gene_frob / gene_norm if gene_norm > 0 else float("inf")

                cell_scores.append(cell_score)
                gene_scores.append(gene_score)

            return {
                'rna_cell_frobenius_mean': np.mean(cell_scores),
                'rna_cell_frobenius_std': np.std(cell_scores),
                'rna_gene_frobenius_mean': np.mean(gene_scores),
                'rna_gene_frobenius_std': np.std(gene_scores)
            }

        except Exception as e:
            print(f"    ❌ RNA预测Frobenius距离计算失败: {e}")
            return self._empty_frobenius_metrics("rna")

    def evaluate_adt_frobenius_distances(self, adt_data, method: str, n_seeds=10,
                                        n_cells=1000, n_proteins=50) -> Dict[str, float]:
        """评估ADT预测的细胞和蛋白Frobenius距离"""
        print(f"  评估ADT预测Frobenius距离...")

        try:
            pred_original = adt_data.uns.get(f'{method}_adt_predicted', None)
            if pred_original is None:
                return self._empty_frobenius_metrics("adt")

            # CLR标准化
            if method == 'connect':
                pred_data = pred_original  # 已经是CLR
            else:
                pred_data = apply_clr_to_prediction(pred_original)

            # 对真实数据进行CLR标准化
            adt_clr = clr_normalize_seurat_style(adt_data, inplace=True)

            cell_scores, pro_scores = [], []

            for seed in range(n_seeds):
                np.random.seed(seed)

                # 抽样
                cell_indices = np.random.choice(adt_clr.shape[0], n_cells, replace=False)
                pro_indices = np.random.choice(adt_clr.shape[1], n_proteins, replace=False)

                # 真实数据
                true_cell_data = adt_clr.X[cell_indices, :]
                true_pro_data = adt_clr.X[:, pro_indices]

                if issparse(true_cell_data):
                    true_cell_data = true_cell_data.toarray()
                if issparse(true_pro_data):
                    true_pro_data = true_pro_data.toarray()

                true_cell_corr = compute_correlation_matrix(true_cell_data)
                true_pro_corr = compute_correlation_matrix(true_pro_data.T)

                cell_norm = np.linalg.norm(true_cell_corr, 'fro')
                pro_norm = np.linalg.norm(true_pro_corr, 'fro')

                # 预测数据
                pred_cell_data = pred_data[cell_indices, :]
                pred_pro_data = pred_data[:, pro_indices]

                pred_cell_corr = compute_correlation_matrix(pred_cell_data)
                pred_pro_corr = compute_correlation_matrix(pred_pro_data.T)

                # 计算Frobenius距离
                cell_frob = compute_frobenius_distance(true_cell_corr, pred_cell_corr)
                pro_frob = compute_frobenius_distance(true_pro_corr, pred_pro_corr)

                cell_score = cell_frob / cell_norm if cell_norm > 0 else float("inf")
                pro_score = pro_frob / pro_norm if pro_norm > 0 else float("inf")

                cell_scores.append(cell_score)
                pro_scores.append(pro_score)

            return {
                'adt_cell_frobenius_mean': np.mean(cell_scores),
                'adt_cell_frobenius_std': np.std(cell_scores),
                'adt_pro_frobenius_mean': np.mean(pro_scores),
                'adt_pro_frobenius_std': np.std(pro_scores)
            }

        except Exception as e:
            print(f"    ❌ ADT预测Frobenius距离计算失败: {e}")
            return self._empty_frobenius_metrics("adt")

    def evaluate_single_method(self, dataset_name: str, method: str) -> Dict[str, Any]:
        """评估单个方法的全部指标"""
        print(f"\n{'='*60}")
        print(f"评估数据集: {dataset_name}, 方法: {method}")
        print(f"{'='*60}")

        start_time = time.time()

        # 加载数据
        data_info = self.load_data(dataset_name)
        if data_info is None:
            return self._empty_result(dataset_name, method, "Failed to load data")

        rna_data = data_info['rna_data']
        adt_data = data_info['adt_data']

        # 初始化结果
        result = {
            'dataset': dataset_name,
            'method': method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 评估5个方面
        metrics = {}

        # 1. Mapping指标
        mapping_metrics = self.evaluate_mapping_metrics(rna_data, adt_data, method)
        metrics.update(mapping_metrics)

        # 2. RNA预测细胞级别指标
        rna_per_cell_metrics = self.evaluate_rna_prediction_per_cell(rna_data, method)
        metrics.update(rna_per_cell_metrics)

        # 3. RNA预测Frobenius距离
        rna_frob_metrics = self.evaluate_rna_frobenius_distances(rna_data, method)
        metrics.update(rna_frob_metrics)

        # 4. ADT预测细胞级别指标
        adt_per_cell_metrics = self.evaluate_adt_prediction_per_cell(adt_data, method)
        metrics.update(adt_per_cell_metrics)

        # 5. ADT预测Frobenius距离
        adt_frob_metrics = self.evaluate_adt_frobenius_distances(adt_data, method)
        metrics.update(adt_frob_metrics)

        result.update(metrics)

        end_time = time.time()
        print(f"\n✅ {method} 评估完成，耗时: {end_time - start_time:.2f} 秒")

        return result

    def save_results_append(self, result: Dict[str, Any]):
        """以追加方式保存结果"""
        # 转换为DataFrame
        df = pd.DataFrame([result])

        # 如果文件不存在，创建并写入表头
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file, index=False)
            print(f"✅ 创建新文件并保存结果: {self.output_file}")
        else:
            # 追加模式，不写入表头
            df.to_csv(self.output_file, mode='a', header=False, index=False)
            print(f"✅ 结果已追加到: {self.output_file}")

    def _empty_mapping_metrics(self) -> Dict[str, float]:
        """返回空的mapping指标"""
        return {
            'foscttm_rna_mean': None, 'foscttm_rna_std': None,
            'foscttm_adt_mean': None, 'foscttm_adt_std': None,
            'rna_area_normalized': None, 'peak_area_normalized': None
        }

    def _empty_per_cell_metrics(self, modality: str) -> Dict[str, float]:
        """返回空的细胞级别指标"""
        prefix = f"{modality}_per_cell"
        return {
            f'{prefix}_r2_mean': None, f'{prefix}_r2_std': None,
            f'{prefix}_correlation_mean': None, f'{prefix}_correlation_std': None,
            f'{prefix}_mse_mean': None, f'{prefix}_mse_std': None,
            f'{prefix}_mae_mean': None, f'{prefix}_mae_std': None,
            f'{prefix}_cv_ratio_mean': None, f'{prefix}_cv_ratio_std': None,
            f'{prefix}_wass_mean': None, f'{prefix}_wass_std': None
        }

    def _empty_frobenius_metrics(self, modality: str) -> Dict[str, float]:
        """返回空的Frobenius指标"""
        cell_key = f"{modality}_cell_frobenius"
        other_key = f"{modality}_gene_frobenius" if modality == "rna" else f"{modality}_pro_frobenius"

        return {
            f'{cell_key}_mean': None, f'{cell_key}_std': None,
            f'{other_key}_mean': None, f'{other_key}_std': None
        }

    def _empty_result(self, dataset_name: str, method: str, error_msg: str) -> Dict[str, Any]:
        """返回空结果"""
        result = {
            'dataset': dataset_name,
            'method': method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # 添加所有指标为None
        for col in self.columns[3:]:  # 跳过前3个基础列
            result[col] = None
        result['status'] = 'failed'
        result['error'] = error_msg
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Integrated evaluation for single-cell multi-omic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 评估特定数据集的特定方法
    python integrated_evaluation.py --dataset issaacseq --method midas

    # 评估特定数据集的多个方法
    python integrated_evaluation.py --dataset issaacseq --method midas --method totalvi

    # 评估多个数据集的特定方法
    python integrated_evaluation.py --dataset issaacseq --dataset SNAREseq_mouse --method midas

    # 指定输出文件
    python integrated_evaluation.py --dataset issaacseq --method midas --output my_results.csv
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        action='append',
        required=True,
        help='Dataset name(s) to evaluate (can be specified multiple times)'
    )

    parser.add_argument(
        '--method',
        type=str,
        action='append',
        required=True,
        help='Method name(s) to evaluate (can be specified multiple times)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='integrated_evaluation_results.csv',
        help='Output CSV file name (default: integrated_evaluation_results.csv)'
    )

    args = parser.parse_args()

    try:
        evaluator = IntegratedMultiModalEvaluator(args.output)

        total_evaluations = len(args.dataset) * len(args.method)
        current_evaluation = 0

        for dataset_name in args.dataset:
            for method_name in args.method:
                current_evaluation += 1
                print(f"\n🔄 进度: {current_evaluation}/{total_evaluations}")

                result = evaluator.evaluate_single_method(dataset_name, method_name)
                evaluator.save_results_append(result)

        print(f"\n🎉 所有评估完成！")
        print(f"📁 结果文件: {evaluator.output_file}")
        print(f"📊 共完成 {total_evaluations} 次评估")

        return 0

    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())