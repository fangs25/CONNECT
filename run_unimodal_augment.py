"""CONNECT workflow with single-modality augmentation.

This script follows the backup experiment setting:

1. standard paired-data training;
2. augmentation with additional single-modality cells;
3. final latent-space alignment.
"""

from os.path import join

import scanpy as sc

from connect import (
    align_model,
    build_model,
    build_paired_loader,
    extract_if_needed,
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
#    Auto-extract from split parts if the .h5ad files are not found.
rna_train = sc.read_h5ad(extract_if_needed(join(data_dir, "rna_train.h5ad"), logger))
adt_train = sc.read_h5ad(extract_if_needed(join(data_dir, "adt_train.h5ad"), logger))
rna_test = sc.read_h5ad(extract_if_needed(join(data_dir, "rna_test.h5ad"), logger))
adt_test = sc.read_h5ad(extract_if_needed(join(data_dir, "adt_test.h5ad"), logger))
unimodal_rna = sc.read_h5ad(extract_if_needed(unimodal_data_path, logger))

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
