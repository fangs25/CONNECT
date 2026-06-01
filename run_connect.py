import os
from os.path import join
import argparse

import scanpy as sc

from connect import (
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
)

parser = argparse.ArgumentParser(description="Run CONNECT standard paired-data training.")
parser.add_argument(
    "--data-dir",
    required=True,
    help=(
        "Directory containing rna_train.h5ad, adt_train.h5ad, "
        "rna_test.h5ad, and adt_test.h5ad."
    ),
)
parser.add_argument("--save-dir", default="./connect_result", help="Directory for CONNECT outputs.")
parser.add_argument("--device", default="0", help="GPU index exposed through CUDA_VISIBLE_DEVICES.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir

set_seed(args.seed)
device = get_device(args.device)
logger = init_logger(save_dir)

# 1. Load paired training and test data.
rna_train = sc.read_h5ad(join(data_dir, "rna_train.h5ad"))
adt_train = sc.read_h5ad(join(data_dir, "adt_train.h5ad"))
rna_test = sc.read_h5ad(join(data_dir, "rna_test.h5ad"))
adt_test = sc.read_h5ad(join(data_dir, "adt_test.h5ad"))
print(rna_train.X.max(), adt_train.X.max())

# 2. Modality-specific preprocessing.
train_rna = make_modality(rna_train, "RNA", preprocess=False)
train_adt = make_modality(adt_train, "ADT", preprocess=True)
test_rna = make_modality(rna_test, "RNA", preprocess=False)
test_adt = make_modality(adt_test, "ADT", preprocess=True)

# 3. Build loaders and model.
data_loader, valid_data_loader = build_paired_loader(
    train_rna,
    train_adt,
    batch_size=128,
    validation_split=0.1,
    num_workers=0,
    training=True,
    logger=logger,
)

model = build_model(data_loader, train_rna, train_adt, device=device)
logger.info(model)

# 4. Standard paired-data training only.
model = train_model(
    model,
    data_loader,
    valid_data_loader,
    device=device,
    epochs=80,
    lr=1e-3,
    weight_decay=1e-3,
    amsgrad=True,
    logger=logger,
)

# 5. Predict and save outputs.
outputs = predict(
    model,
    test_rna,
    test_adt,
    device=device,
    batch_size=128,
    num_workers=2,
    logger=logger,
)

save_model(model, save_dir)
save_outputs(outputs, save_dir)

rna_test.obsm['connect_rna_emb'] = outputs['mod1_latent'].detach().cpu().numpy()
rna_test.obsm['connect_rna_mapped'] = outputs['mod1_mapping_from_mod2'].detach().cpu().numpy()
rna_test.obsm['connect_rna_predicted'] = outputs['mod1_predicted_from_mod2'].detach().cpu().numpy()
 
adt_test.obsm['connect_adt_emb'] = outputs['mod2_latent'].detach().cpu().numpy()
adt_test.obsm['connect_adt_mapped'] = outputs['mod2_mapping_from_mod1'].detach().cpu().numpy()
adt_test.obsm['connect_adt_predicted'] = outputs['mod2_predicted_from_mod1'].detach().cpu().numpy()

