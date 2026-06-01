Installation
============

Clone the repository and install the scientific Python dependencies used by
your experiments.

.. code-block:: bash

   git clone <your-repository-url>
   cd <your-repository>
   pip install -r docs/requirements.txt

The model itself requires the usual single-cell and PyTorch stack, including
``torch``, ``scanpy``, ``anndata``, ``scipy``, ``scikit-learn``, and ``pandas``.
Install the CUDA-enabled PyTorch build that matches your machine.

For deterministic CUDA behavior, the tutorials set:

.. code-block:: python

   import os
   os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
