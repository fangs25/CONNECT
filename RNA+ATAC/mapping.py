#!/usr/bin/env python3
"""
Mapping Metrics Calculator - RNA+ATAC Version
专门用于RNA+ATAC模态的mapping指标计算
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

try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps


# ==================== 复制的函数 ====================

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


# ==================== 主要计算类 ====================

class RNATACMappingMetricsCalculator:
    """RNA+ATAC模态的Mapping指标计算器"""

    def __init__(self):
        self.output_dir = "mapping_results"
        self.modality = "RNA+ATAC"
        self.ensure_output_directories()

        print(f"RNA+ATAC Mapping Metrics Calculator 初始化完成")
        print(f"输出目录: {self.output_dir}")

    def ensure_output_directories(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "foscttm"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "topk_accuracy"), exist_ok=True)

    def load_rna_atac_mapping_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载RNA+ATAC模态的mapping数据
        """
        print(f"加载 {dataset_name} 的RNA+ATAC mapping数据...")

        try:


            # rna_data = sc.read_h5ad(rna_path)
            rna_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/anndata/{dataset_name}/rna_test.h5ad')
            print(f"✓ RNA数据加载成功: {rna_data.shape}")
            # atac_data = sc.read_h5ad(atac_path)
            atac_data = sc.read_h5ad(f'/DATA2/zhangjingxiao/fangs/models_2025/Model_results/anndata/{dataset_name}/atac_test.h5ad')

            print(f"✓ ATAC数据加载成功: {atac_data.shape}")

            # 获取可用的mapping方法
            available_methods = []
            for key in rna_data.uns.keys():
                if key.endswith('_rna_mapped'):
                    method = key.replace('_rna_mapped', '')
                    if f'{method}_atac_mapped' in atac_data.uns:
                        available_methods.append(method)

            print(f"✓ 可用mapping方法: {available_methods}")

            return {
                'rna_data': rna_data,
                'atac_data': atac_data,
                'available_methods': available_methods,
                'dataset_name': dataset_name
            }

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None

    def calculate_rna_atac_method_metrics(self, rna_data, atac_data, method: str) -> Dict[str, Any]:
        """
        计算RNA+ATAC模态单个方法的mapping指标
        """
        print(f"\n计算RNA+ATAC方法: {method}")

        try:
            # 获取嵌入和映射结果
            rna_emb = rna_data.uns.get(f'{method}_rna_emb', None)
            atac_emb = atac_data.uns.get(f'{method}_atac_emb', None)
            rna_mapped = rna_data.uns.get(f'{method}_rna_mapped', None)
            atac_mapped = atac_data.uns.get(f'{method}_atac_mapped', None)

            if rna_mapped is None or atac_mapped is None:
                print(f'  ❌ 缺少RNA+ATAC映射结果')
                return None

            # 转换为numpy数组
            if hasattr(rna_emb, 'numpy'):
                rna_emb = rna_emb.numpy()
            if hasattr(atac_emb, 'numpy'):
                atac_emb = atac_emb.numpy()
            if hasattr(rna_mapped, 'numpy'):
                rna_mapped = rna_mapped.numpy()
            if hasattr(atac_mapped, 'numpy'):
                atac_mapped = atac_mapped.numpy()

            print(f'  RNA嵌入形状: {rna_emb.shape}')
            print(f'  ATAC嵌入形状: {atac_emb.shape}')
            print(f'  RNA映射形状: {rna_mapped.shape}')
            print(f'  ATAC映射形状: {atac_mapped.shape}')

            # 使用全部数据，不进行抽样
            print(f'  使用全部 {rna_emb.shape[0]} 个细胞计算RNA+ATAC mapping指标')

            # 计算FOSCTTM
            print(f'  📊 计算FOSCTTM (RNA+ATAC)...')
            foscttm_atac_full = FOSFTTM_batch(atac_mapped, atac_emb, output_full=True)
            foscttm_rna_full = FOSFTTM_batch(rna_mapped, rna_emb, output_full=True)

            print(f'  ATAC FOSCTTM: 均值={np.mean(foscttm_atac_full):.4f}, 标准差={np.std(foscttm_atac_full):.4f}')
            print(f'  RNA FOSCTTM: 均值={np.mean(foscttm_rna_full):.4f}, 标准差={np.std(foscttm_rna_full):.4f}')

            # 计算Top-K Accuracy
            print(f'  📊 计算Top-K Accuracy (RNA+ATAC)...')

            # 计算不同k值
            n_cells = rna_emb.shape[0]
            rate_list = np.arange(0.5, 10.1, 0.5)  # 0.5%, 1%, ... 10%
            rate_based_k = [int(n_cells * rate * 0.01) for rate in rate_list]
            fixed_k = [1, 2, 3, 4, 5, 8, 10]
            all_k_values = sorted(set(rate_based_k + fixed_k))
            all_k_values = [k for k in all_k_values if k < n_cells and k > 0]

            print(f'  评估 {len(all_k_values)} 个k值')

            # 评估每个k值的准确率
            top_k_results = []
            for k in all_k_values:
                k_peak = top_k_matching_accuracy_hnsw(atac_mapped, atac_emb, k)
                k_rna = top_k_matching_accuracy_hnsw(rna_mapped, rna_emb, k)

                top_k_results.append({
                    'k_value': k,
                    'peak_accuracy': k_peak,
                    'rna_accuracy': k_rna
                })

            # 计算曲线下面积
            current_results_df = pd.DataFrame(top_k_results)
            current_results_df = current_results_df.sort_values(by='k_value')

            peak_area = simps(current_results_df['peak_accuracy'], current_results_df['k_value'])
            rna_area = simps(current_results_df['rna_accuracy'], current_results_df['k_value'])

            max_k = max(all_k_values)
            peak_area_normalized = peak_area / max_k
            rna_area_normalized = rna_area / max_k

            print(f'  Peak归一化面积: {peak_area_normalized:.4f}')
            print(f'  RNA归一化面积: {rna_area_normalized:.4f}')

            return {
                'foscttm_atac_full': foscttm_atac_full.tolist(),
                'foscttm_rna_full': foscttm_rna_full.tolist(),
                'foscttm_atac_mean': np.mean(foscttm_atac_full),
                'foscttm_atac_median': np.median(foscttm_atac_full),
                'foscttm_atac_std': np.std(foscttm_atac_full),
                'foscttm_rna_mean': np.mean(foscttm_rna_full),
                'foscttm_rna_median': np.median(foscttm_rna_full),
                'foscttm_rna_std': np.std(foscttm_rna_full),
                'peak_area_normalized': peak_area_normalized,
                'rna_area_normalized': rna_area_normalized,
                'top_k_data': top_k_results,
                'n_cells': n_cells
            }

        except Exception as e:
            print(f"  ❌ 计算RNA+ATAC方法 {method} 失败: {e}")
            return None

    def calculate_rna_atac_dataset_metrics(self, dataset_name: str) -> Dict[str, Any]:
        """
        计算单个数据集的RNA+ATAC模态mapping指标
        """
        print(f"\n{'='*80}")
        print(f"计算RNA+ATAC数据集: {dataset_name}")
        print(f"{'='*80}")

        start_time = time.time()

        # 加载数据
        data_info = self.load_rna_atac_mapping_data(dataset_name)
        if data_info is None:
            return {'dataset': dataset_name, 'modality': self.modality, 'status': 'failed', 'error': 'Failed to load data'}

        rna_data = data_info['rna_data']
        atac_data = data_info['atac_data']
        available_methods = data_info['available_methods']

        if not available_methods:
            return {'dataset': dataset_name, 'modality': self.modality, 'status': 'failed', 'error': 'No mapping methods available'}

        results = {
            'dataset': dataset_name,
            'modality': self.modality,
            'n_cells': rna_data.shape[0],
            'methods': {},
            'status': 'completed'
        }

        # 为每个方法计算指标
        for method in available_methods:
            method_results = self.calculate_rna_atac_method_metrics(rna_data, atac_data, method)
            if method_results:
                results['methods'][method] = method_results
            else:
                results['methods'][method] = {'status': 'failed'}

        # 保存结果
        self.save_rna_atac_foscttm_csv(dataset_name, results)
        self.save_rna_atac_topk_json(dataset_name, results)

        end_time = time.time()
        results['processing_time'] = end_time - start_time

        print(f"\n✅ RNA+ATAC数据集 {dataset_name} 计算完成，耗时: {results['processing_time']:.2f} 秒")

        return results

    def save_rna_atac_foscttm_csv(self, dataset_name: str, results: Dict[str, Any]):
        """
        保存RNA+ATAC模态FOSCTTM细胞级别数据到CSV文件
        """
        foscttm_data = []

        # 获取细胞数量
        n_cells = results['n_cells']

        # # 为每个细胞创建记录
        for cell_id in range(n_cells):
            row = {'cell_id': cell_id}

            for method, method_data in results['methods'].items():
                if method_data.get('status') != 'failed' and 'foscttm_atac_full' in method_data:
                    # 使用ATAC方向的FOSCTTM值
                    if cell_id < len(method_data['foscttm_atac_full']):
                        row[method] = method_data['foscttm_atac_full'][cell_id]
                    else:
                        row[method] = None
                else:
                    row[method] = None

            foscttm_data.append(row)

        # 创建DataFrame
        df = pd.DataFrame(foscttm_data)

        # 保存到文件
        output_file = os.path.join(self.output_dir, "foscttm", f"{dataset_name}_rna_to_atac_foscttm.csv")
        df.to_csv(output_file, index=False)

                # 为每个细胞创建记录
        for cell_id in range(n_cells):
            row = {'cell_id': cell_id}

            for method, method_data in results['methods'].items():
                if method_data.get('status') != 'failed' and 'foscttm_rna_full' in method_data:
                    # 使用ATAC方向的FOSCTTM值
                    if cell_id < len(method_data['foscttm_rna_full']):
                        row[method] = method_data['foscttm_rna_full'][cell_id]
                    else:
                        row[method] = None
                else:
                    row[method] = None

            foscttm_data.append(row)

        # 创建DataFrame
        df = pd.DataFrame(foscttm_data)

        # 保存到文件
        output_file = os.path.join(self.output_dir, "foscttm", f"{dataset_name}_atac_to_rna_foscttm.csv")
        df.to_csv(output_file, index=False)

        print(f"  ✓ RNA+ATAC FOSCTTM数据已保存: {output_file}")
        print(f"    形状: {df.shape} (细胞 x 方法)")

    def save_rna_atac_topk_json(self, dataset_name: str, results: Dict[str, Any]):
        """
        保存RNA+ATAC模态Top-K Accuracy数据到JSON文件
        """
        topk_data = {}

        for method, method_data in results['methods'].items():
            if method_data.get('status') != 'failed' and 'top_k_data' in method_data:
                # 提取k-value和对应的准确率
                topk_records = method_data['top_k_data']

                k_values = []
                peak_accuracies = []
                rna_accuracies = []

                for record in topk_records:
                    k_values.append(record['k_value'])
                    peak_accuracies.append(record['peak_accuracy'])
                    rna_accuracies.append(record['rna_accuracy'])

                topk_data[method] = {
                    'k_values': k_values,
                    'peak_accuracies': peak_accuracies,
                    'rna_accuracies': rna_accuracies,
                    'peak_area': method_data.get('peak_area_normalized', 0),
                    'rna_area': method_data.get('rna_area_normalized', 0)
                }

        # 保存到文件
        output_file = os.path.join(self.output_dir, "topk_accuracy", f"{dataset_name}_rna_atac_topk_accuracy.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(topk_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ RNA+ATAC Top-K数据已保存: {output_file}")
        print(f"    包含 {len(topk_data)} 个方法的结果")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate RNA+ATAC mapping metrics from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
            # 计算特定数据集
            python mapping_metrics_calculator.py --dataset issaacseq

            # 计算多个数据集
            python mapping_metrics_calculator.py --dataset issaacseq --dataset SNAREseq_mouse
                    """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        action='append',
        help='RNA+ATAC dataset name(s) to calculate (can be specified multiple times)'
    )

    args = parser.parse_args()

    if not args.dataset:
        print("错误: 必须指定 --dataset")
        return 1

    try:
        calculator = RNATACMappingMetricsCalculator()

        for dataset_name in args.dataset:
            calculator.calculate_rna_atac_dataset_metrics(dataset_name)

        print(f"\n📁 输出目录: {calculator.output_dir}")
        print(f"   - RNA+ATAC FOSCTTM数据: {os.path.join(calculator.output_dir, 'foscttm/')}")
        print(f"   - RNA+ATAC Top-K数据: {os.path.join(calculator.output_dir, 'topk_accuracy/')}")

        print(f"\n✅ RNA+ATAC Mapping指标计算完成，无图片生成！")

        return 0

    except Exception as e:
        print(f"❌ 计算过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())