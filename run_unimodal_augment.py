"""CONNECT workflow with single-modality augmentation.

This script follows the backup experiment setting:

1. standard paired-data training;
2. augmentation with additional single-modality cells;
3. final latent-space alignment.
"""

import os
from os.path import join

import scanpy as sc

from connect import (
    align_model,
    build_model,
    build_paired_loader,
    get_device,
    init_logger,
    make_modality,
    predict,
    save_model,
    save_outputs,
    set_seed,
    train_model,
    train_with_unimodal,
)

save_dir = "./connect_unimodal_augment_result"
data_dir = "./data/RNA_ADT/RPE008"
unimodal_data_path = "./data/RNA_ADT/RNA_only/carotid_rna.h5ad"

set_seed(42)
device = get_device("0")
logger = init_logger(save_dir)

# 1. Load paired data and additional single-modality data.
rna_train = sc.read_h5ad(join(data_dir, "rna_train.h5ad"))
adt_train = sc.read_h5ad(join(data_dir, "adt_train.h5ad"))
rna_test = sc.read_h5ad(join(data_dir, "rna_test.h5ad"))
adt_test = sc.read_h5ad(join(data_dir, "adt_test.h5ad"))
unimodal_rna = sc.read_h5ad(unimodal_data_path)

# 2. Describe modality-specific preprocessing.
train_rna = make_modality(rna_train, "RNA", preprocess=False)
train_adt = make_modality(adt_train, "ADT", preprocess=True)
test_rna = make_modality(rna_test, "RNA", preprocess=False)
test_adt = make_modality(adt_test, "ADT", preprocess=True)

# 3. Build loaders and model from paired data.
data_loader, valid_data_loader = build_paired_loader(
    train_rna,
    train_adt,
    batch_size=128,
    validation_split=0.1,
    num_workers=2,
    training=True,
    logger=logger,
)

model = build_model(data_loader, train_rna, train_adt, device=device)
# logger.info(model)

# 4. Stage 1: standard paired-data training.
model = train_model(
    model,
    data_loader,
    valid_data_loader,
    device=device,
    epochs=80,
    lr=1e-3,
    weight_decay=1e-3,
    amsgrad=False,
    logger=logger,
)

# 5. Stage 2: single-modality augmentation.
model = train_with_unimodal(
    model,
    data_loader,
    unimodal=unimodal_rna,
    unimodal_type="RNA",
    unimodal_preprocess=False,
    valid_loader=valid_data_loader,
    device=device,
    epochs=10,
    batch_size=128,
    num_workers=2,
    lr=1e-3,
    weight_decay=1e-3,
    amsgrad=False,
    logger=logger,
)

# 6. Stage 3: latent-space alignment.
model = align_model(
    model,
    data_loader,
    valid_loader=valid_data_loader,
    device=device,
    epochs=10,
    lr=5e-4,
    weight_decay=1e-3,
    amsgrad=False,
    logger=logger,
)

# 7. Predict and save outputs.
outputs = predict(
    model,
    test_rna,
    test_adt,
    device=device,
    batch_size=128,
    num_workers=2,
    logger=logger,
)

# save_model(model, save_dir)
# save_outputs(outputs, save_dir)
rna_test.uns['connect_rna_emb'] = outputs['mod1_latent'].detach().cpu().numpy()
rna_test.uns['connect_rna_mapped'] = outputs['mod1_mapping_from_mod2'].detach().cpu().numpy()
rna_test.uns['connect_rna_predicted'] = outputs['mod1_predicted_from_mod2'].detach().cpu().numpy()
 

adt_test.uns['connect_adt_emb'] = outputs['mod2_latent'].detach().cpu().numpy()
adt_test.uns['connect_adt_mapped'] = outputs['mod2_mapping_from_mod1'].detach().cpu().numpy()
adt_test.uns['connect_adt_predicted'] = outputs['mod2_predicted_from_mod1'].detach().cpu().numpy()

from integration import *
dataset_name = 'RPE008'
method_name = 'connect'


rna_data = rna_test
adt_original = adt_test
adt_ = adt_original.copy()
adt_data = clr_normalize_seurat_style(adt_, inplace=True)    

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

data_info =  {
        'rna_data': rna_data,
        'adt_data': adt_data,
        'available_methods': available_methods,
        'dataset_name': dataset_name
    }



evaluator = IntegratedMultiModalEvaluator('integrated_evaluation_results.csv')
evaluator.data_info = data_info
result = evaluator.evaluate_single_method(dataset_name, method_name)
print(result)
