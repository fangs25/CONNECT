Data loading
============

Paired data
-----------

.. autoclass:: connect.dataloader.MultiomicsDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: connect.dataloader.CellDataset
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing helpers
---------------------

.. autofunction:: connect.dataloader.run_tfidf_optimized

.. autofunction:: connect.dataloader.clr_normalize_seurat_style

Single-modality augmentation
----------------------------

.. autoclass:: connect.dataloader.AugmentSingleDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: connect.dataloader.AugmentSingleCellDataset
   :members:
   :undoc-members:
   :show-inheritance:

Batch collation
---------------

.. autofunction:: connect.dataloader.sparse_collate_fn

.. autofunction:: connect.dataloader.sparse_collate_fn_single

Base loader
-----------

.. autoclass:: connect.dataloader.BaseDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

Logging helper
--------------

.. autofunction:: connect.dataloader.get_safe_logger
