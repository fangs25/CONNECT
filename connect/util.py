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
