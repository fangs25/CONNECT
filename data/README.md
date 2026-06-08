## Datasets

### RPE008 — RNA+ADT paired dataset (included)

The `data/RPE008/` directory contains a real RNA+ADT paired single-cell
dataset (RPE008 cell line) ready for use with CONNECT.

Because several `.h5ad` files exceed GitHub's 20 MB file-size limit, they are
stored as **split parts**:

```text
data/RPE008/
  rna_train.h5ad.part_aa … rna_train.h5ad.part_au   (21 parts, ~19 MB each)
  rna_test.h5ad.part_aa … rna_test.h5ad.part_ac     (3 parts)
  adt_train.h5ad.part_aa … adt_train.h5ad.part_ab   (2 parts)
  adt_test.h5ad.part_aa                              (1 part)
```

The CONNECT run scripts (`run_connect.py` and `run_unimodal_augment.py`)
**automatically reassemble** these parts on first use.  You do not need to
manually concatenate anything — simply point `--data-dir` at
`./data/RPE008` and the scripts will detect the missing `.h5ad` files and
reconstruct them from the parts.

If you need to reassemble manually for any reason:

```bash
cd data/RPE008
cat rna_train.h5ad.part_* > rna_train.h5ad
cat rna_test.h5ad.part_*  > rna_test.h5ad
cat adt_train.h5ad.part_* > adt_train.h5ad
cat adt_test.h5ad.part_*  > adt_test.h5ad
```

### Published datasets (Zenodo)

Additional datasets used in CONNECT experiments (RNA+ATAC, other cell lines,
and single-modality data) can be downloaded from
[Zenodo](https://doi.org/10.5281/zenodo.20490832).

Expected data organisation for your own datasets:

```text
data/
  RNA_ADT/
    RPE008/
      rna_train.h5ad
      adt_train.h5ad
      rna_test.h5ad
      adt_test.h5ad
  RNA_ATAC/
    .../
      rna_train.h5ad
      atac_train.h5ad
      rna_test.h5ad
      atac_test.h5ad
```

Input files should be AnnData objects (`.h5ad`) with cells in rows and features
in columns.
