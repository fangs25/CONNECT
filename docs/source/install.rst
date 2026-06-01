Installation
============

Clone the repository and install the scientific Python dependencies used by
your experiments.

.. code-block:: bash

   git clone https://github.com/fangs25/CONNECT.git
   cd CONNECT
   conda env create -f environment.yml
   conda activate connect-env


The model itself requires the usual single-cell and PyTorch stack, including
``torch``, ``scanpy``, ``anndata``, ``scipy``, ``scikit-learn``, and ``pandas``.
Install the CUDA-enabled PyTorch build that matches your machine. For example:

.. code-block:: bash
   
   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
   # or 
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
