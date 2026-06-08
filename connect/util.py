import glob
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from itertools import repeat

def inf_loop(data_loader):
    """Yield batches from a dataloader indefinitely.

    Parameters
    ----------
    data_loader
        Iterable PyTorch dataloader.

    Yields
    ------
    object
        Mini-batches produced by ``data_loader``.  When one pass finishes, the
        generator restarts from the beginning.
    """
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    """Track running averages for named scalar metrics."""

    def __init__(self, *keys, writer=None):
        """Initialize metric storage and an optional scalar writer.

        Parameters
        ----------
        *keys
            Initial metric names to track.
        writer
            Optional object with an ``add_scalar`` method, for example a
            TensorBoard writer.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """Reset totals, counts, and averages to zero."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """Add a new metric observation.

        Parameters
        ----------
        key
            Metric name.
        value
            Scalar value to add.
        n
            Number of observations represented by ``value``.
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # self._data.total[key] += value * n
        # self._data.counts[key] += n
        # self._data.average[key] = self._data.total[key] / self._data.counts[key]

        if key not in self._data.index:
            self._data.loc[key] = [0, 0, 0]  # [total, counts, average]
    
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        """Return the running average for one metric.

        Parameters
        ----------
        key
            Metric name.

        Returns
        -------
        float
            Current average for ``key``.
        """
        # return self._data.average[key]
        return self._data.loc[key, 'average']

    def result(self):
        """Return all metric averages as a dictionary.

        Returns
        -------
        dict
            Mapping from metric names to their running averages.
        """
        return dict(self._data.average)


def extract_if_needed(h5ad_path: str, logger=None) -> str:
    """Ensure an .h5ad file exists, reassembling from split parts if necessary.

    CONNECT datasets that exceed GitHub's 20 MB file-size limit are stored as
    split parts (``<name>.h5ad.part_aa``, ``<name>.h5ad.part_ab``, …).  This
    helper checks whether the target ``.h5ad`` file already exists on disk.  If
    it does, the path is returned unchanged.  Otherwise the parts are
    concatenated in alphabetical order to reconstruct the original file.

    Parameters
    ----------
    h5ad_path
        Path to the expected ``.h5ad`` file.
    logger
        Optional logger for progress messages.

    Returns
    -------
    str
        The original ``h5ad_path`` — either it already existed or it was just
        reassembled from parts.

    Raises
    ------
    FileNotFoundError
        If neither the ``.h5ad`` file nor any split parts are found.
    """
    path = Path(h5ad_path)
    if path.exists():
        return str(path)

    # Look for split parts in the same directory.
    part_pattern = str(path.parent / (path.name + ".part_*"))
    parts = sorted(glob.glob(part_pattern))

    if not parts:
        raise FileNotFoundError(
            f"Neither {path} nor split parts ({path.name}.part_*) were found. "
            f"Please ensure the dataset is available in {path.parent}."
        )

    if logger:
        logger.info(f"Reassembling {path.name} from {len(parts)} split part(s) ...")
    else:
        print(f"Reassembling {path.name} from {len(parts)} split part(s) ...",
              file=sys.stderr)

    with open(path, "wb") as out_f:
        for part_path in parts:
            with open(part_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)

    if logger:
        logger.info(f"Reassembled {path.name} ({path.stat().st_size:,} bytes).")
    else:
        print(f"Reassembled {path.name} ({path.stat().st_size:,} bytes).",
              file=sys.stderr)

    return str(path)
