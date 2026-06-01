CONNECT documentation
=====================

``CONNECT`` is a lightweight PyTorch workflow for paired single-cell
multi-omic data.  The package trains modality-specific autoencoders, aligns
their latent spaces, and generates cross-modal predictions.

The documentation is organized around executable-style notebooks.  Each
tutorial follows the same explicit workflow:

1. load ``AnnData`` objects or ``.h5ad`` files;
2. describe modality-specific preprocessing;
3. build paired data loaders;
4. construct and train the model;
5. infer embeddings, mappings, and cross-modal predictions.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   install
   tutorials/index
   api/index
