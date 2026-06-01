import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, issparse, diags
import scipy.sparse as sp
from tqdm import tqdm
import logging
import torch
import anndata
import scanpy as sc


class BaseDataLoader(DataLoader):
    """Base DataLoader with deterministic train/validation splitting."""

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, drop_last = False):
        """Create a DataLoader and optionally attach train/validation samplers.

        Parameters
        ----------
        dataset
            PyTorch dataset instance to wrap.
        batch_size
            Number of samples returned by each mini-batch.
        shuffle
            Whether to shuffle samples when no validation sampler is used.
        validation_split
            Validation split definition.  Use ``0.0`` to disable validation, a
            float in ``(0, 1)`` for a fraction of the dataset, or an integer for
            an exact number of validation samples.
        num_workers
            Number of subprocesses used by PyTorch for data loading.
        collate_fn
            Function used to combine individual dataset items into a batch.
        drop_last
            Whether to drop the last incomplete batch.
        """
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last':drop_last,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        """Create deterministic subset samplers for train and validation splits.

        Parameters
        ----------
        split
            Same convention as ``validation_split`` in :meth:`__init__`.

        Returns
        -------
        tuple
            ``(train_sampler, valid_sampler)``.  Both are ``None`` when
            validation is disabled.
        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        # np.random.seed(0)
        np.random.seed(42)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        """Return a validation DataLoader that shares initialization settings.

        Returns
        -------
        torch.utils.data.DataLoader or None
            Validation loader when ``validation_split`` is non-zero; otherwise
            ``None``.
        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


def get_safe_logger(logger=None, name="null"):
    """Return a provided logger or a NullHandler-backed fallback logger.

    Parameters
    ----------
    logger
        Existing logger object.  If provided, it is returned unchanged.
    name
        Name of the fallback logger created when ``logger`` is ``None``.

    Returns
    -------
    logging.Logger
        Logger that can be safely used without configuring global logging.
    """
    if logger is not None:
        return logger
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


class MultiomicsDataLoader(BaseDataLoader):
    """DataLoader for paired cells from two modalities."""

    def __init__(self, 
            modality_1,
            modality_2,
            batch_size = 256,
            shuffle=True, 
            validation_split=0.0, 
            num_workers=1, 
            training=True,
            logger = None,
            drop_last = True,
            ):
        """Construct the paired dataset and wrap it in the base loader.

        Parameters
        ----------
        modality_1, modality_2
            Modality dictionaries accepted by :class:`CellDataset`.  Each
            dictionary must contain ``"data"``, ``"type"``, and
            ``"preprocess"`` fields.
        batch_size
            Number of paired cells per mini-batch.
        shuffle
            Whether to shuffle cells in the training loader.
        validation_split
            Fraction or count of cells reserved for validation.
        num_workers
            Number of PyTorch data-loading worker processes.
        training
            Whether the loader is used for training.  In inference mode, raw
            count matrices are cached on the dataset.
        logger
            Optional logger for preprocessing messages.
        drop_last
            Whether to drop the final incomplete batch.
        """
        
        self.dataset = CellDataset(modality_1, modality_2, training = training, logger = logger)


        super().__init__(self.dataset, batch_size = batch_size, shuffle = shuffle, 
                            validation_split = validation_split, num_workers = num_workers, collate_fn = sparse_collate_fn, drop_last = drop_last)


class CellDataset(Dataset):
    """Paired-cell dataset backed by AnnData h5ad files.

    Each sample returns processed inputs and raw-count targets:
    ``(modality_1, modality_2, modality_1_raw, modality_2_raw)``.
    """
    def __init__(
        self,
        modality_1,
        modality_2,
        training = True,
        logger = None,
    ):
        """Load both modalities and cache processed and raw matrices.

        Parameters
        ----------
        modality_1, modality_2
            Dictionaries describing paired data.  ``"data"`` can be an
            :class:`anndata.AnnData` object or a path to an ``.h5ad`` file.
            ``"type"`` must be one of ``"RNA"``, ``"ATAC"``, or ``"ADT"``.
            ``"preprocess"`` controls modality-specific preprocessing; for
            ATAC this can be a TF-IDF method such as ``"Stuart"`` or ``"10x"``.
            The optional ``"ceiling"`` flag clips sparse counts to ``[0, 4]``.
        training
            Whether the dataset is used for training.  If ``False``, dense raw
            count arrays are stored as attributes for downstream inference.
        logger
            Optional logger for status messages.
        """
        super().__init__()
        self.training = training

        self.logger = get_safe_logger(logger)

        assert modality_1['type'] in ['RNA', 'ATAC', 'ADT'], f'Modality must be RNA, ATAC or ADT, but {modality_1["type"]} found.'
        assert modality_2['type'] in ['RNA', 'ATAC', 'ADT'], f'Modality must be RNA, ATAC or ADT, but {modality_2["type"]} found.'

        self.modality_1_type = modality_1['type']
        self.modality_2_type = modality_2['type']

        self.modality_1_data = self.preprocess(modality_1)
        self.modality_2_data = self.preprocess(modality_2)
        self.modality_1_data_raw = self.preprocess({**modality_1, "preprocess": False}, info = False)
        self.modality_2_data_raw = self.preprocess({**modality_2, "preprocess": False}, info = False)
        
        self.modality_1_dim, self.modality_2_dim = self.modality_1_data.shape[1], self.modality_2_data.shape[1]
        assert self.modality_1_data.shape[0] == self.modality_2_data.shape[0], 'Please input paired data with the same first dimension during training.'

        self.logger.info(f'Modality 1 : {modality_1["type"]} with dimension {self.modality_1_data.shape}; Preprocess : {modality_1["preprocess"]}')
        self.logger.info(f'Modality 2 : {modality_2["type"]} with dimension {self.modality_2_data.shape}; Preprocess : {modality_2["preprocess"]}')
    
    def preprocess(self, modality, info = True):
        """Read one modality and apply RNA, ATAC, or ADT preprocessing.

        Parameters
        ----------
        modality
            Modality dictionary with ``data``, ``type``, ``preprocess``, and
            optional ``ceiling`` entries.
        info
            Whether to emit file-reading and clipping messages.

        Returns
        -------
        scipy.sparse.csr_matrix
            Processed feature matrix with cells in rows and features in columns.
        """
        
        adata = modality['data']
        modality_type = modality['type']
        prepro = modality['preprocess']
        ceiling = modality.get('ceiling', False)

        if isinstance(adata, anndata.AnnData):
            adata_uniomic = adata.copy()
        elif isinstance(adata, str):
            if info:
                self.logger.info(f'Reading {modality_type} from H5AD files...')
            adata_uniomic = sc.read_h5ad(adata)
        else:
            raise ValueError('Please input Anndata or data directory')
        adata_uniomic.var_names_make_unique()

        # Ensure data is in CSR format
        if not isspmatrix_csr(adata_uniomic.X):
            self.logger.info(f'Converting {modality_type} to CSR format...')
            adata_uniomic.X = csr_matrix(adata_uniomic.X.astype(np.int32))

        # Truncate the sparse matrix data
        if ceiling:
            if info:
                self.logger.info(f'Truncating {modality_type} counts...')
            adata_uniomic.X.data = np.clip(adata_uniomic.X.data, 0, 4)

        # Store raw counts for inference mode
        if not self.training:
            raw_counts_name = f'{modality_type.lower()}_raw_counts'
            setattr(self, raw_counts_name, adata_uniomic.X.toarray().astype(np.float32))

        # Apply modality-specific preprocessing
        if prepro:
            if modality_type == 'RNA':
                self.logger.info('Preprocessing RNA...')
                sc.pp.normalize_total(adata_uniomic, target_sum=1e4)
                sc.pp.log1p(adata_uniomic)
                processed_X = adata_uniomic.X

            elif modality_type == 'ATAC':
                self.logger.info('Preprocessing ATAC...')
                adata_uniomic = run_tfidf_optimized(adata_uniomic, method = prepro, chunk_size=0, logger = self.logger)  # 'Stuart', 'Cusanovich' 'Andrew' '10x'

                processed_X = csr_matrix(adata_uniomic.X.astype(np.float32))

            elif modality_type == 'ADT':
                self.logger.info('Preprocessing ADT...')
                adata_uniomic = clr_normalize_seurat_style(adata_uniomic)
                processed_X = csr_matrix(adata_uniomic.X.astype(np.float32))

            # Ensure consistency in matrix format
            if not isspmatrix_csr(processed_X):
                processed_X = csr_matrix(processed_X)
            # self.adata_uniomic_X = processed_X
            adata_uniomic_X = processed_X

        else:
            # self.adata_uniomic_X = adata_uniomic.X
            adata_uniomic_X = adata_uniomic.X

        return adata_uniomic_X
    
    def __len__(self):
        """Return the number of paired cells."""
        return self.modality_1_data.shape[0]

    def __getitem__(self, idx):
        """Return processed inputs and raw targets for one paired cell.

        Parameters
        ----------
        idx
            Integer row index.

        Returns
        -------
        tuple of torch.Tensor
            ``(modality_1_processed, modality_2_processed,
            modality_1_raw, modality_2_raw)``.
        """
        modality_1_data = self.modality_1_data[idx].toarray().astype(np.float32) 
        modality_2_data = self.modality_2_data[idx].toarray().astype(np.float32)
        modality_1_data_raw = self.modality_1_data_raw[idx].toarray().astype(np.float32) 
        modality_2_data_raw = self.modality_2_data_raw[idx].toarray().astype(np.float32)  

        return torch.tensor(modality_1_data), torch.tensor(modality_2_data), torch.tensor(modality_1_data_raw), torch.tensor(modality_2_data_raw)


class AugmentSingleDataLoader(BaseDataLoader):
    """DataLoader for optional single-modality augmentation data."""

    def __init__(self, 
            modality,
            batch_size = 256,
            shuffle=True, 
            validation_split=0.0, 
            num_workers=1, 
            training=True,
            logger = None,
            drop_last = True,
            ):
        """Construct the single-modality dataset used for augmentation.

        Parameters
        ----------
        modality
            Modality dictionary accepted by :class:`AugmentSingleCellDataset`.
        batch_size
            Number of cells per mini-batch.
        shuffle
            Whether to shuffle cells.
        validation_split
            Fraction or count of cells reserved for validation.
        num_workers
            Number of PyTorch data-loading worker processes.
        training
            Whether raw count arrays should be cached for inference.
        logger
            Optional logger for preprocessing messages.
        drop_last
            Whether to drop the final incomplete batch.
        """
        
        self.dataset = AugmentSingleCellDataset(modality, training = training, logger = logger)


        super().__init__(self.dataset, batch_size = batch_size, shuffle = shuffle, 
                            validation_split = validation_split, num_workers = num_workers, collate_fn = sparse_collate_fn_single, drop_last = drop_last)


class AugmentSingleCellDataset(Dataset):
    """Single-modality dataset returning processed inputs and raw targets."""

    def __init__(
        self,
        modality,
        training = True,
        logger = None,
    ):
        """Load the single-modality matrix and its raw-count counterpart.

        Parameters
        ----------
        modality
            Dictionary describing one modality.  ``"data"`` can be an AnnData
            object or h5ad path, ``"type"`` must be ``"RNA"``, ``"ATAC"``, or
            ``"ADT"``, and ``"preprocess"`` controls normalization.
        training
            Whether the dataset is used during training.
        logger
            Optional logger for status messages.
        """
        super().__init__()
        self.training = training

        self.logger = get_safe_logger(logger)

        assert modality['type'] in ['RNA', 'ATAC', 'ADT'], f'Modality must be RNA, ATAC or ADT, but {modality["type"]} found.'

        self.modality_1_type = modality['type']

        self.modality_1_data = self.preprocess(modality)
        self.modality_1_data_raw = self.preprocess({**modality, "preprocess": False}, info = False)
        
        self.modality_1_dim = self.modality_1_data.shape[1]

        self.logger.info(f'Modality 1 : {modality["type"]} with dimension {self.modality_1_data.shape}; Preprocess : {modality["preprocess"]}')
    
    def preprocess(self, modality, info = True):
        """Read and preprocess a single modality for augmentation.

        Parameters
        ----------
        modality
            Modality dictionary with ``data``, ``type``, ``preprocess``, and
            optional ``ceiling`` entries.
        info
            Whether to log file-reading and clipping messages.

        Returns
        -------
        scipy.sparse.csr_matrix
            Processed feature matrix.
        """
        
        adata = modality['data']
        modality_type = modality['type']
        prepro = modality['preprocess']
        ceiling = modality.get('ceiling', False)

        if isinstance(adata, anndata.AnnData):
            adata_uniomic = adata.copy()
        elif isinstance(adata, str):
            if info:
                self.logger.info(f'Reading {modality_type} from H5AD files...')
            adata_uniomic = sc.read_h5ad(adata)
        else:
            raise ValueError('Please input Anndata or data directory')
        adata_uniomic.var_names_make_unique()

        # Ensure data is in CSR format
        if not isspmatrix_csr(adata_uniomic.X):
            self.logger.info(f'Converting {modality_type} to CSR format...')
            adata_uniomic.X = csr_matrix(adata_uniomic.X.astype(np.int32))

        # Truncate the sparse matrix data
        if ceiling:
            if info:
                self.logger.info(f'Truncating {modality_type} counts...')
            adata_uniomic.X.data = np.clip(adata_uniomic.X.data, 0, 4)

        # Store raw counts for inference mode
        if not self.training:
            raw_counts_name = f'{modality_type.lower()}_raw_counts'
            setattr(self, raw_counts_name, adata_uniomic.X.toarray().astype(np.float32))

        # Apply modality-specific preprocessing
        if prepro:
            if modality_type == 'RNA':
                self.logger.info('Preprocessing RNA...')
                sc.pp.normalize_total(adata_uniomic, target_sum=1e4)
                sc.pp.log1p(adata_uniomic)
                processed_X = adata_uniomic.X

            elif modality_type == 'ATAC':
                self.logger.info('Preprocessing ATAC...')
                adata_uniomic = run_tfidf_optimized(adata_uniomic, method = prepro, chunk_size=0, logger = self.logger)  # 'Stuart', 'Cusanovich' 'Andrew' '10x'

                processed_X = csr_matrix(adata_uniomic.X.astype(np.float32))

            elif modality_type == 'ADT':
                self.logger.info('Preprocessing ADT...')
                adata_uniomic = clr_normalize_seurat_style(adata_uniomic)
                processed_X = csr_matrix(adata_uniomic.X.astype(np.float32))

            # Ensure consistency in matrix format
            if not isspmatrix_csr(processed_X):
                processed_X = csr_matrix(processed_X)
            # self.adata_uniomic_X = processed_X
            adata_uniomic_X = processed_X

        else:
            # self.adata_uniomic_X = adata_uniomic.X
            adata_uniomic_X = adata_uniomic.X

        return adata_uniomic_X
    
    def __len__(self):
        """Return the number of cells in the single-modality dataset."""
        return self.modality_1_data.shape[0]

    def __getitem__(self, idx):
        """Return processed input and raw target for one cell.

        Parameters
        ----------
        idx
            Integer row index.

        Returns
        -------
        tuple of torch.Tensor
            ``(processed_input, raw_target)``.
        """
        modality_1_data = self.modality_1_data[idx].toarray().astype(np.float32) 
        modality_1_data_raw = self.modality_1_data_raw[idx].toarray().astype(np.float32) 

        return torch.tensor(modality_1_data), torch.tensor(modality_1_data_raw)


def clr_normalize_seurat_style(adata, inplace=True):
    """Apply Seurat-style centered log-ratio normalization to ADT data.

    Parameters
    ----------
    adata
        AnnData object whose ``X`` matrix contains raw or count-like ADT
        features with cells in rows and proteins in columns.
    inplace
        If ``True``, modify ``adata`` directly.  If ``False``, work on a copy.

    Returns
    -------
    anndata.AnnData
        AnnData object with normalized values stored in ``X`` as ``float32``.
    """

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


def sparse_collate_fn(batch):
    """Collate paired-modality samples into dense tensor batches.

    Parameters
    ----------
    batch
        List of tuples returned by :class:`CellDataset`.

    Returns
    -------
    tuple of torch.Tensor
        ``(modality_1_processed, modality_2_processed,
        modality_1_raw, modality_2_raw)``.
    """
    rna_batch, atac_batch, rna_batch_raw, atac_batch_raw = zip(*batch)  # 解包 batch

    # 将 batch 内的所有数据转换为 Tensor，并拼接
    rna_batch = torch.cat(rna_batch, dim=0)  # 形状 (batch_size, num_genes)
    atac_batch = torch.cat(atac_batch, dim=0)  # 形状 (batch_size, num_peaks)

    rna_batch_raw = torch.cat(rna_batch_raw, dim=0)  # 形状 (batch_size, num_genes)
    atac_batch_raw = torch.cat(atac_batch_raw, dim=0)  # 形状 (batch_size, num_peaks)

    return rna_batch, atac_batch, rna_batch_raw, atac_batch_raw


def sparse_collate_fn_single(batch):
    """Collate single-modality augmentation samples.

    Parameters
    ----------
    batch
        List of tuples returned by :class:`AugmentSingleCellDataset`.

    Returns
    -------
    tuple of torch.Tensor
        ``(processed_input, raw_target)``.
    """
    batch, batch_raw = zip(*batch)  # 解包 batch

    # 将 batch 内的所有数据转换为 Tensor，并拼接
    batch = torch.cat(batch, dim=0)  # 形状 (batch_size, num_genes)

    batch_raw = torch.cat(batch_raw, dim=0)  # 形状 (batch_size, num_genes)

    return batch, batch_raw, 


def run_tfidf_optimized(adata: anndata.AnnData, 
                        method: str = 'Stuart', 
                        scale_factor: float = 1e4, 
                        layer: str = None,
                        inplace: bool = True,
                        chunk_size: int = 5000,
                        logger = None) -> anndata.AnnData:
    """Apply memory-aware TF-IDF normalization to ATAC data in AnnData.
    
    Parameters
    ----------
    adata
        AnnData object with cells as rows and peaks as columns.
    method
        TF-IDF variant.  Supported values are ``"Stuart"``,
        ``"Cusanovich"``, ``"Andrew"``, and ``"10x"``.
    scale_factor
        Scaling factor applied before log transformation for methods that use
        scaled term frequency.
    layer
        Optional AnnData layer to read instead of ``adata.X``.
    inplace
        Whether to modify ``adata`` directly.  If ``False``, a copy is returned.
    chunk_size
        Number of rows or columns processed per chunk.  Set to ``0`` to disable
        chunking.
    logger
        Optional logger for progress messages.
    
    Returns
    -------
    anndata.AnnData
        AnnData object with TF-IDF normalized matrix stored in ``adata.X`` when
        ``layer`` is ``None``.  When ``layer`` is provided, results are stored in
        ``adata.layers[f"X_tfidf_{method}"]``.
    """
    if not inplace:
        adata = adata.copy()

    log = get_safe_logger(logger)

    log.info(f'TF-IDF using {method} method with optimized memory usage...')
    X = adata.X if layer is None else adata.layers[layer]
    
    if not sp.issparse(X):
        X = csr_matrix(X)
    
    n_cells, n_peaks = X.shape
    
    # ------------------
    # Step 1: Compute TF
    # ------------------
    if method != '10x':
        log.info("Computing row sums...")
        # 使用分块计算行和，避免一次性加载所有数据
        row_sums = np.zeros(n_cells, dtype=np.float32)
        if chunk_size > 0 and n_cells > chunk_size:
            for i in tqdm(range(0, n_cells, chunk_size), desc="Processing rows"):
                chunk = X[i:i+chunk_size]
                row_sums[i:i+chunk_size] = chunk.sum(axis=1).A1
        else:
            row_sums = X.sum(axis=1).A1
        
        # 避免除以零
        row_sums[row_sums == 0] = 1.0
        row_factors = 1 / row_sums
        
        # 使用稀疏对角矩阵乘法代替广播
        log.info("Normalizing TF...")
        diag_matrix = diags(row_factors, format='csr')
        tf = diag_matrix @ X
    else:
        # Method 10x don't normalize
        tf = X.astype(np.float32)

    # ------------------
    # Step 2: Compute IDF
    # ------------------
    log.info("Computing document frequencies...")
    
    # 使用更高效的方法计算非零元素
    if chunk_size > 0 and n_peaks > chunk_size:
        doc_freq = np.zeros(n_peaks, dtype=np.int32)
        for j in tqdm(range(0, n_peaks, chunk_size), desc="Processing columns"):
            chunk = X[:, j:j+chunk_size]
            doc_freq[j:j+chunk_size] = (chunk > 0).sum(axis=0).A1
    else:
        # 使用getnnz方法避免创建临时密集矩阵
        doc_freq = X.getnnz(axis=0)
    
    doc_freq = np.maximum(doc_freq, 1)  # 等价于clip但更快
    idf = n_cells / doc_freq
    
    # Method-specific IDF process
    if method in ['Cusanovich', 'Andrew']:
        idf = np.log1p(idf)

    # ------------------
    # Step 3: Compute TF-IDF
    # ------------------
    log.info("Computing TF-IDF...")
    
    if method == 'Stuart':
        # log(scale_factor * TF * IDF)
        # 使用对角矩阵乘法避免广播
        idf_diag = diags(idf, format='csr')
        tf_idf = tf @ idf_diag
        
        # 只对非零元素应用log变换
        tf_idf = tf_idf * scale_factor
        if sp.issparse(tf_idf):
            tf_idf.data = np.log1p(tf_idf.data)
        else:
            tf_idf = np.log1p(tf_idf)
            
    elif method == 'Cusanovich':
        # TF * log(IDF)
        idf_diag = diags(idf, format='csr')
        tf_idf = tf @ idf_diag
        
    elif method == 'Andrew':
        # log(scale_factor*TF) * log(IDF)
        # 只对非零元素应用log变换
        log_tf = tf.copy()
        if sp.issparse(log_tf):
            log_tf.data = np.log1p(log_tf.data * scale_factor)
        else:
            log_tf = np.log1p(log_tf * scale_factor)
            
        log_idf = np.log1p(idf)
        idf_diag = diags(log_idf, format='csr')
        tf_idf = log_tf @ idf_diag
        
    elif method == '10x':
        # TF * IDF (no TF normalization)
        idf_diag = diags(idf, format='csr')
        tf_idf = tf @ idf_diag
        
    else:
        raise ValueError(f"Invalid method: {method}. Choose: 'Stuart', 'Cusanovich' 'Andrew' '10x'")

    # 确保使用32位浮点数节省内存
    if sp.issparse(tf_idf):
        tf_idf = tf_idf.astype(np.float32)
    else:
        tf_idf = tf_idf.astype(np.float32)
    
    # 存储结果
    output_layer = f"X_tfidf_{method}" if layer else "X"
    if layer:
        adata.layers[output_layer] = tf_idf
    else:
        adata.X = tf_idf
    
    return adata
