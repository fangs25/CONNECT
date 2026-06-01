"""Step-by-step helpers for running CONNECT.

This module intentionally keeps the workflow as explicit function calls.  Each function
corresponds to one visible workflow step: set seed, describe modalities, build
loaders, build model, train, predict, and save outputs.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .dataloader import AugmentSingleDataLoader, MultiomicsDataLoader
from .logger import setup_logging
from .model import MultiModalityAE
from .trainer import AlignTrainer, AugmentTrainer, Trainer


def set_seed(seed: int = 42) -> None:
    """Set random seeds used by NumPy, PyTorch, and CuDNN.

    This function should be called before data loader construction and model
    initialization when reproducible results are required.

    Parameters
    ----------
    seed
        Integer seed used by ``torch.manual_seed`` and ``numpy.random.seed``.

    Returns
    -------
    None
        The function updates global random states in place.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "0") -> torch.device:
    """Select a CUDA device by index string, falling back to CPU when needed.

    Parameters
    ----------
    device
        GPU index exposed through ``CUDA_VISIBLE_DEVICES``.  For example,
        ``"0"`` selects the first visible GPU.  If CUDA is unavailable, the
        returned device is ``torch.device("cpu")``.

    Returns
    -------
    torch.device
        ``cuda:0`` when CUDA is available after setting ``CUDA_VISIBLE_DEVICES``;
        otherwise ``cpu``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def as_device(device) -> torch.device:
    """Convert a device string or ``torch.device`` into a ``torch.device``.

    Parameters
    ----------
    device
        A device specification such as ``"cpu"``, ``"cuda:0"``, or an existing
        ``torch.device`` object.

    Returns
    -------
    torch.device
        Normalized PyTorch device object.
    """
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def init_logger(save_dir: str = "./connect_result") -> logging.Logger:
    """Create the output directory and initialize console/file logging.

    Parameters
    ----------
    save_dir
        Directory where the log file ``info.log`` and downstream model outputs
        are written.  The directory is created if it does not exist.

    Returns
    -------
    logging.Logger
        Logger named ``"train"`` configured by :func:`connect.logger.setup_logging`.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path)
    return logging.getLogger("train")


def make_modality(data, modality_type: str, preprocess=False, ceiling: bool = False) -> Dict:
    """Describe one modality for CONNECT data loading.

    Parameters
    ----------
    data
        Path to an h5ad file or an AnnData object.
    modality_type
        One of ``"RNA"``, ``"ADT"``, or ``"ATAC"``.
    preprocess
        ``False`` for no transform, ``True`` for RNA/ADT defaults, or a TF-IDF
        method string such as ``"Andrew"`` for ATAC.
    ceiling
        Whether to truncate sparse count values before preprocessing.

    Returns
    -------
    dict
        Dictionary with keys ``"type"``, ``"data"``, ``"preprocess"``, and
        ``"ceiling"``.  This dictionary is passed to
        :class:`connect.dataloader.MultiomicsDataLoader`.

    Notes
    -----
    Typical settings are:

    - RNA: ``make_modality(rna, "RNA", preprocess=False)``
    - ADT: ``make_modality(adt, "ADT", preprocess=True)``
    - ATAC: ``make_modality(atac, "ATAC", preprocess="Andrew", ceiling=True)``
    """
    return {
        "type": modality_type,
        "data": data,
        "preprocess": preprocess,
        "ceiling": ceiling,
    }


def build_paired_loader(
    modality_1: Dict,
    modality_2: Dict,
    batch_size: int = 128,
    validation_split: float = 0.1,
    num_workers: int = 2,
    training: bool = True,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[MultiomicsDataLoader, Optional[torch.utils.data.DataLoader]]:
    """Build paired-data loaders after applying modality-specific preprocessing.

    Parameters
    ----------
    modality_1
        Modality dictionary for the first branch, usually RNA, created by
        :func:`make_modality`.
    modality_2
        Modality dictionary for the second branch, usually ADT or ATAC, created
        by :func:`make_modality`.
    batch_size
        Number of paired cells per mini-batch.
    validation_split
        Fraction of cells used as validation data when ``training=True``.
        Set to ``0.0`` to disable validation.
    num_workers
        Number of worker processes used by ``torch.utils.data.DataLoader``.
        Use ``0`` during debugging when exact batch order needs to be inspected.
    training
        Whether the loader is used for training.  Training loaders can be
        shuffled and split into train/validation subsets.  Prediction loaders
        should use ``training=False``.
    shuffle
        Whether to shuffle sampled cells.  If ``None``, defaults to
        ``training``.
    drop_last
        Whether to drop the final incomplete batch.  If ``None``, defaults to
        ``training``.
    logger
        Optional logger used by the dataset preprocessing code.

    Returns
    -------
    tuple
        ``(data_loader, valid_data_loader)``.  ``valid_data_loader`` is ``None``
        when ``validation_split=0`` or ``training=False``.
    """
    shuffle = training if shuffle is None else shuffle
    drop_last = training if drop_last is None else drop_last
    loader = MultiomicsDataLoader(
        modality_1,
        modality_2,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_split=validation_split if training else 0.0,
        num_workers=num_workers,
        training=training,
        drop_last=drop_last,
        logger=logger,
    )
    return loader, loader.split_validation()


def build_model(
    train_loader: MultiomicsDataLoader,
    modality_1: Dict,
    modality_2: Dict,
    device: torch.device = torch.device("cpu"),
    adaptive_weight: bool = False,
) -> MultiModalityAE:
    """Construct the CONNECT model from the dimensions in the training loader.

    Parameters
    ----------
    train_loader
        Paired training data loader returned by :func:`build_paired_loader`.
        Feature dimensions are read from ``train_loader.dataset``.
    modality_1
        Modality dictionary for the first branch.  Its ``"type"`` field is used
        to determine the modality pair.
    modality_2
        Modality dictionary for the second branch.  Supported pairs are
        RNA+ADT and RNA+ATAC.
    device
        Device on which the model is allocated, e.g. ``"cuda:0"`` or
        ``torch.device("cuda:0")``.
    adaptive_weight
        If ``True``, loss weights are learned through trainable logits.  The
        default ``False`` uses the fixed weights defined by the CONNECT model.

    Returns
    -------
    connect.model.MultiModalityAE
        Initialized two-branch CONNECT model.
    """
    device = as_device(device)
    n_genes = train_loader.dataset.modality_1_dim
    n_regions = train_loader.dataset.modality_2_dim if modality_2["type"] == "ATAC" else 0
    n_proteins = train_loader.dataset.modality_2_dim if modality_2["type"] == "ADT" else 0
    return MultiModalityAE(
        ae_modality_1=modality_1,
        ae_modality_2=modality_2,
        n_genes=n_genes,
        n_regions=n_regions,
        n_proteins=n_proteins,
        adaptive_weight=adaptive_weight,
        device=device,
    )


def make_optimizer(model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 1e-3, amsgrad: bool = True):
    """Create AdamW using currently trainable parameters only.

    Parameters
    ----------
    model
        Model whose parameters are optimized.  Parameters with
        ``requires_grad=False`` are ignored.
    lr
        Learning rate for AdamW.
    weight_decay
        Weight decay coefficient for AdamW.
    amsgrad
        Whether to use the AMSGrad variant of AdamW.

    Returns
    -------
    torch.optim.AdamW
        Optimizer over trainable model parameters.
    """
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters are available.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)


def train_model(
    model: MultiModalityAE,
    train_loader: MultiomicsDataLoader,
    valid_loader=None,
    device: torch.device = torch.device("cpu"),
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    amsgrad: bool = True,
    logger: Optional[logging.Logger] = None,
) -> MultiModalityAE:
    """Train CONNECT on paired data using the standard training stage.

    The standard stage optimizes four objectives: within-modality
    reconstruction, cross-modality prediction, latent-space mapping, and
    contrastive matching between paired cells.

    Parameters
    ----------
    model
        CONNECT model returned by :func:`build_model`.
    train_loader
        Paired training data loader.
    valid_loader
        Optional validation data loader returned by :func:`build_paired_loader`.
    device
        Device used for training.
    epochs
        Number of standard training epochs.
    lr
        Learning rate for AdamW.
    weight_decay
        Weight decay coefficient for AdamW.
    amsgrad
        Whether AdamW uses the AMSGrad update.
    logger
        Optional logger for progress messages.

    Returns
    -------
    connect.model.MultiModalityAE
        The trained model.  The object is updated in place and also returned for
        convenient chaining.
    """
    device = as_device(device)
    for parameter in model.parameters():
        parameter.requires_grad = True
    trainer = Trainer(
        model,
        make_optimizer(model, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad),
        device=device,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        epochs=epochs,
        logger=logger,
    )
    trainer.train()
    return model


def train_with_unimodal(
    model: MultiModalityAE,
    train_loader: MultiomicsDataLoader,
    unimodal,
    unimodal_type: str = "RNA",
    unimodal_preprocess=False,
    valid_loader=None,
    device: torch.device = torch.device("cpu"),
    epochs: int = 10,
    batch_size: int = 128,
    num_workers: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    amsgrad: bool = True,
    logger: Optional[logging.Logger] = None,
) -> MultiModalityAE:
    """Refine the model with additional single-modality cells.

    This is the second stage used in the unimodal augmentation workflow.  The
    current implementation updates the modality-1 encoder/reconstruction path
    and interleaves unimodal reconstruction regularization with paired-data
    training.

    Parameters
    ----------
    model
        Model after standard paired training.
    train_loader
        Paired training data loader reused during augmentation.
    unimodal
        Additional single-modality data, either an AnnData object or path to an
        ``.h5ad`` file.
    unimodal_type
        Type of the unimodal data, e.g. ``"RNA"``, ``"ADT"``, or ``"ATAC"``.
    unimodal_preprocess
        Preprocessing setting for the unimodal data.  Use the same convention as
        :func:`make_modality`.
    valid_loader
        Optional validation loader from the paired training data.
    device
        Device used for training.
    epochs
        Number of augmentation epochs.
    batch_size
        Batch size for the unimodal data loader.
    num_workers
        Number of workers for the unimodal data loader.
    lr
        Learning rate for AdamW.
    weight_decay
        Weight decay coefficient for AdamW.
    amsgrad
        Whether AdamW uses the AMSGrad update.
    logger
        Optional logger for progress messages.

    Returns
    -------
    connect.model.MultiModalityAE
        Model updated by the unimodal augmentation stage.
    """
    device = as_device(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "ae_modality_1.encoder" in name or "ae_modality_1.decoder_recon" in name

    unimodal_loader = AugmentSingleDataLoader(
        make_modality(unimodal, unimodal_type, unimodal_preprocess),
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=num_workers,
        training=True,
        drop_last=True,
        logger=logger,
    )
    trainer = AugmentTrainer(
        model,
        make_optimizer(model, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad),
        device=device,
        epochs=epochs,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        unimodal_loader=unimodal_loader,
        unimodal_optimizer=make_optimizer(model, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad),
        logger=logger,
    )
    trainer.train()
    return model


def align_model(
    model: MultiModalityAE,
    train_loader: MultiomicsDataLoader,
    valid_loader=None,
    device: torch.device = torch.device("cpu"),
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    amsgrad: bool = True,
    logger: Optional[logging.Logger] = None,
) -> MultiModalityAE:
    """Fine-tune latent mappings after paired or augmented training.

    This optional stage optimizes latent alignment and isometry terms, with a
    small contrastive component, to improve cross-modality matching.

    Parameters
    ----------
    model
        Model after standard training or unimodal augmentation.
    train_loader
        Paired training data loader.
    valid_loader
        Optional validation loader.
    device
        Device used for training.
    epochs
        Number of alignment epochs.
    lr
        Learning rate for AdamW.  A smaller value such as ``5e-4`` is commonly
        used for this stage.
    weight_decay
        Weight decay coefficient for AdamW.
    amsgrad
        Whether AdamW uses the AMSGrad update.
    logger
        Optional logger for progress messages.

    Returns
    -------
    connect.model.MultiModalityAE
        Model updated by the alignment stage.
    """
    device = as_device(device)
    for parameter in model.parameters():
        parameter.requires_grad = True
    trainer = AlignTrainer(
        model,
        make_optimizer(model, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad),
        device=device,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        epochs=epochs,
        logger=logger,
    )
    trainer.train()
    return model


def predict(
    model: MultiModalityAE,
    modality_1: Dict,
    modality_2: Dict,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """Run CONNECT inference and return embeddings plus cross-modal predictions.

    Parameters
    ----------
    model
        Trained CONNECT model.
    modality_1
        Test modality dictionary for branch 1.
    modality_2
        Test modality dictionary for branch 2.
    device
        Device used for inference.
    batch_size
        Batch size for prediction.
    num_workers
        Number of DataLoader workers.
    logger
        Optional logger used by test data preprocessing.

    Returns
    -------
    dict
        Dictionary of CPU tensors with keys:

        ``"mod1_latent"``
            Latent embedding of modality 1.
        ``"mod2_latent"``
            Latent embedding of modality 2.
        ``"mod2_predicted_from_mod1"``
            Predicted modality-2 features from modality-1 input.
        ``"mod1_predicted_from_mod2"``
            Predicted modality-1 features from modality-2 input.
        ``"mod2_mapping_from_mod1"``
            Modality-1 latent mapped into modality-2 latent space.
        ``"mod1_mapping_from_mod2"``
            Modality-2 latent mapped into modality-1 latent space.
    """
    device = as_device(device)
    loader, _ = build_paired_loader(
        modality_1,
        modality_2,
        batch_size=batch_size,
        validation_split=0.0,
        num_workers=num_workers,
        training=False,
        shuffle=False,
        drop_last=False,
        logger=logger,
    )
    model.eval()
    outputs = defaultdict(list)
    with torch.no_grad():
        for mod1_arr, mod2_arr, _, _ in tqdm(loader, desc="Inference"):
            mod1_arr = mod1_arr.to(device)
            mod2_arr = mod2_arr.to(device)
            (
                mod1_latent,
                mod2_latent,
                _,
                _,
                mod2_predicted_from_mod1,
                mod1_predicted_from_mod2,
                mod2_mapping_from_mod1,
                mod1_mapping_from_mod2,
            ) = model.forward(mod1_arr, mod2_arr)
            outputs["mod1_latent"].append(mod1_latent.cpu())
            outputs["mod2_latent"].append(mod2_latent.cpu())
            outputs["mod2_predicted_from_mod1"].append(mod2_predicted_from_mod1.cpu())
            outputs["mod1_predicted_from_mod2"].append(mod1_predicted_from_mod2.cpu())
            outputs["mod2_mapping_from_mod1"].append(mod2_mapping_from_mod1.cpu())
            outputs["mod1_mapping_from_mod2"].append(mod1_mapping_from_mod2.cpu())
    return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def save_model(model: MultiModalityAE, save_dir: str = "./connect_result", filename: str = "model_state.pt") -> Path:
    """Save model weights and return the saved path.

    Parameters
    ----------
    model
        Trained CONNECT model.
    save_dir
        Directory where the model checkpoint is written.
    filename
        Checkpoint file name.

    Returns
    -------
    pathlib.Path
        Full path to the saved checkpoint.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    path = save_path / filename
    torch.save(model.state_dict(), path)
    return path


def save_outputs(outputs: Dict[str, torch.Tensor], save_dir: str = "./connect_result", filename: str = "model_outputs.pt") -> Path:
    """Save inference outputs and return the saved path.

    Parameters
    ----------
    outputs
        Dictionary returned by :func:`predict`.
    save_dir
        Directory where output tensors are written.
    filename
        Output file name.

    Returns
    -------
    pathlib.Path
        Full path to the saved output tensor file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    path = save_path / filename
    torch.save(outputs, path)
    return path
