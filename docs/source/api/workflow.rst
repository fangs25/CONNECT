Workflow
========

The workflow module provides the user-facing functions used in the example
scripts.  These helpers keep the main pipeline explicit: prepare input
modalities, create dataloaders, build the model, train, infer, and save results.

Standard paired-data workflow
-----------------------------

.. autofunction:: connect.workflow.make_modality

.. autofunction:: connect.workflow.build_paired_loader

.. autofunction:: connect.workflow.build_model

.. autofunction:: connect.workflow.make_optimizer

.. autofunction:: connect.workflow.train_model

.. autofunction:: connect.workflow.predict

.. autofunction:: connect.workflow.save_outputs

.. autofunction:: connect.workflow.save_model

Optional training stages
------------------------

.. autofunction:: connect.workflow.train_with_unimodal

.. autofunction:: connect.workflow.align_model

Setup helpers
-------------

.. autofunction:: connect.workflow.set_seed

.. autofunction:: connect.workflow.get_device

.. autofunction:: connect.workflow.as_device

.. autofunction:: connect.workflow.init_logger
