"""Neural network modules for CONNECT."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from .loss import MatchLoss

class ContrastiveAE(nn.Module):
    """One modality branch of CONNECT.

    A branch contains an encoder, a decoder for within-modality reconstruction,
    a decoder for cross-modality prediction, and a linear mapping layer that
    projects its latent representation into the paired modality latent space.
    """

    def __init__(
        self,
        num_input_genes,
        num_output_genes,
        device="cuda",
        seed=0,
        latent_dim=32,
        dropout=0.2,
        input_dropout=0.2,
        relu=True,
        hidden_dim=None,
        decoder_pred_hidden=None,
        decoder_recon_hidden=None,
    ):
        """Initialize all submodules for one modality branch.

        Parameters
        ----------
        num_input_genes
            Number of input features for this branch.  For RNA this is the
            number of genes; for ATAC this is the number of regions; for ADT
            this is the number of proteins.
        num_output_genes
            Number of features in the paired modality predicted by this branch.
        device
            Device on which branch parameters are allocated.
        seed
            Stored for reproducibility bookkeeping.  Random seeds should be set
            globally before model construction.
        latent_dim
            Dimension of the shared latent embedding.
        dropout
            Dropout rate used between hidden layers.
        input_dropout
            Dropout rate applied to the branch input before the first hidden
            layer.
        relu
            Whether to apply ReLU to decoder outputs.
        hidden_dim
            Encoder hidden layer sizes.  Defaults to ``[1024, 256]``.
        decoder_pred_hidden
            Hidden layer sizes for the cross-modality prediction decoder.
        decoder_recon_hidden
            Hidden layer sizes for the reconstruction decoder.
        """
        super().__init__()
        hidden_dim = [1024, 256] if hidden_dim is None else hidden_dim
        decoder_pred_hidden = [1024] if decoder_pred_hidden is None else decoder_pred_hidden
        decoder_recon_hidden = [] if decoder_recon_hidden is None else decoder_recon_hidden

        self.num_input_genes = num_input_genes
        self.num_output_genes = num_output_genes
        self.device = device
        self.seed = seed

        self.encoder = Encoder(
            num_input_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_dropout=input_dropout,
        )
        self.decoder_recon = Decoder(
            num_input_genes,
            latent_dim=latent_dim,
            hidden_dim=list(reversed(decoder_recon_hidden)),
            dropout=dropout,
            relu=relu,
        )
        self.mapping = Mapping(latent_dim, latent_dim)
        self.decoder_pred = Decoder(
            num_output_genes,
            latent_dim=latent_dim,
            hidden_dim=list(reversed(decoder_pred_hidden)),
            dropout=dropout,
            relu=relu,
        )

        self.recon_mse = nn.MSELoss(reduction="mean")
        self.pred_mse = nn.MSELoss(reduction="mean")
        self.mapping_loss = nn.MSELoss(reduction="mean")
        self.to(self.device)

    def forward(self, x):
        """Encode one modality and return all branch outputs.

        Parameters
        ----------
        x
            Input tensor with shape ``(n_cells, num_input_genes)``.

        Returns
        -------
        tuple
            ``(latent, reconstruction, prediction, mapped_latent)``.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder_recon(latent)
        prediction = self.decoder_pred(latent)
        mapped_latent = self.mapping(latent)
        return latent, reconstruction, prediction, mapped_latent


class MultiModalityAE(nn.Module):
    """Two-branch CONNECT autoencoder for paired RNA+ATAC or RNA+ADT data.

    The model uses one :class:`ContrastiveAE` branch per modality.  It supports
    paired RNA+ADT and RNA+ATAC training, with modality-specific decoder
    structures, contrastive temperature, and fixed loss weights.
    """

    def __init__(
        self,
        ae_modality_1: Optional[dict] = None,
        ae_modality_2: Optional[dict] = None,
        n_genes: int = 0,
        n_regions: int = 0,
        n_proteins: int = 0,
        adaptive_weight: bool = False,
        device: str = "cuda",
    ):
        """Configure the architecture and loss weights for one modality pair.

        Parameters
        ----------
        ae_modality_1
            Modality dictionary for branch 1.  Usually RNA.  Must include a
            ``"type"`` field.
        ae_modality_2
            Modality dictionary for branch 2.  Usually ADT or ATAC.  Must
            include a ``"type"`` field.
        n_genes
            Number of RNA features.
        n_regions
            Number of ATAC features.  Set to ``0`` when the second modality is
            ADT.
        n_proteins
            Number of ADT features.  Set to ``0`` when the second modality is
            ATAC.
        adaptive_weight
            If ``True``, loss weights are learned with a softmax over trainable
            logits.  If ``False``, fixed modality-pair weights are used.
        device
            Device on which model parameters are allocated.
        """
        super().__init__()
        ae_modality_1 = {} if ae_modality_1 is None else ae_modality_1
        ae_modality_2 = {} if ae_modality_2 is None else ae_modality_2

        assert (n_genes > 0) + (n_regions > 0) + (n_proteins > 0) == 2, "Provide exactly two modalities."

        if ae_modality_1["type"] in ["RNA", "ATAC"] and ae_modality_2["type"] in ["RNA", "ATAC"]:
            self.modality_pair = "RNA_ATAC"
            self.ae_modality_1 = ContrastiveAE(
                relu=True,
                decoder_pred_hidden=[],
                decoder_recon_hidden=[1024],
                num_input_genes=n_genes,
                num_output_genes=n_regions,
                device=device,
            )
            self.ae_modality_2 = ContrastiveAE(
                relu=True,
                decoder_pred_hidden=[1024],
                decoder_recon_hidden=[],
                num_input_genes=n_regions,
                num_output_genes=n_genes,
                device=device,
            )
            self._default_weights = torch.tensor([10, 10, 1, 1], dtype=torch.float32, device=device)
            temperature = 0.2

        elif ae_modality_1["type"] in ["RNA", "ADT"] and ae_modality_2["type"] in ["RNA", "ADT"]:
            self.modality_pair = "RNA_ADT"
            self.ae_modality_1 = ContrastiveAE(
                relu=True,
                decoder_pred_hidden=[1024, 512],
                decoder_recon_hidden=[1024],
                num_input_genes=n_genes,
                num_output_genes=n_proteins,
                device=device,
            )
            self.ae_modality_2 = ContrastiveAE(
                relu=True,
                decoder_pred_hidden=[1024, 512],
                decoder_recon_hidden=[1024],
                num_input_genes=n_proteins,
                num_output_genes=n_genes,
                device=device,
            )
            self._default_weights = torch.tensor([1, 10, 1, 1], dtype=torch.float32, device=device)
            temperature = 0.05
        else:
            raise ValueError("CONNECT expects RNA+ATAC or RNA+ADT paired data.")

        self.adaptive_weight = adaptive_weight
        self.device = device
        self.contrastive_loss = MatchLoss(temperature)
        self.logits = nn.Parameter(torch.zeros(4, device=device))

    @property
    def weights(self) -> torch.Tensor:
        """Return weights for mapping, contrastive, reconstruction, and prediction losses.

        Returns
        -------
        torch.Tensor
            Tensor of length four ordered as ``[w_map, w_cl, w_recon, w_pred]``.
        """
        if self.adaptive_weight:
            return torch.softmax(self.logits, dim=0)
        return self._default_weights.clone()

    def forward(self, mod1_arr, mod2_arr):
        """Run a paired forward pass and return all latent and decoded outputs.

        Parameters
        ----------
        mod1_arr
            Tensor for modality 1 with shape ``(n_cells, n_modality_1_features)``.
        mod2_arr
            Tensor for modality 2 with shape ``(n_cells, n_modality_2_features)``.

        Returns
        -------
        tuple
            ``(mod1_latent, mod2_latent, mod1_recon_from_mod1,
            mod2_recon_from_mod2, mod2_predicted_from_mod1,
            mod1_predicted_from_mod2, mod2_mapping_from_mod1,
            mod1_mapping_from_mod2)``.
        """
        mod1_latent = self.ae_modality_1.encoder(mod1_arr)
        mod2_latent = self.ae_modality_2.encoder(mod2_arr)
        mod2_mapping_from_mod1 = self.ae_modality_1.mapping(mod1_latent)
        mod1_mapping_from_mod2 = self.ae_modality_2.mapping(mod2_latent)

        mod1_recon_from_mod1 = self.ae_modality_1.decoder_recon(mod1_latent)
        mod2_recon_from_mod2 = self.ae_modality_2.decoder_recon(mod2_latent)
        mod2_predicted_from_mod1 = self.ae_modality_1.decoder_pred(mod1_latent)
        mod1_predicted_from_mod2 = self.ae_modality_2.decoder_pred(mod2_latent)

        return (
            mod1_latent,
            mod2_latent,
            mod1_recon_from_mod1,
            mod2_recon_from_mod2,
            mod2_predicted_from_mod1,
            mod1_predicted_from_mod2,
            mod2_mapping_from_mod1,
            mod1_mapping_from_mod2,
        )

    def forward_modality1(self, modality1_arr):
        """Run the modality-1 branch only.

        Parameters
        ----------
        modality1_arr
            Input tensor for modality 1.

        Returns
        -------
        tuple
            ``(latent, reconstruction, cross_modal_prediction, mapped_latent)``.
        """
        latent = self.ae_modality_1.encoder(modality1_arr)
        return (
            latent,
            self.ae_modality_1.decoder_recon(latent),
            self.ae_modality_1.decoder_pred(latent),
            self.ae_modality_1.mapping(latent),
        )

    def forward_modality2(self, modality2_arr):
        """Run the modality-2 branch only.

        Parameters
        ----------
        modality2_arr
            Input tensor for modality 2.

        Returns
        -------
        tuple
            ``(latent, reconstruction, cross_modal_prediction, mapped_latent)``.
        """
        latent = self.ae_modality_2.encoder(modality2_arr)
        return (
            latent,
            self.ae_modality_2.decoder_recon(latent),
            self.ae_modality_2.decoder_pred(latent),
            self.ae_modality_2.mapping(latent),
        )


class Encoder(nn.Module):
    """Multilayer perceptron encoder that returns L2-normalized latent vectors."""

    def __init__(self, n_genes: int, latent_dim: int = 32, hidden_dim: List[int] = None, dropout: float = 0.2, input_dropout: float = 0.2):
        """Create the encoder network.

        Parameters
        ----------
        n_genes
            Number of input features.
        latent_dim
            Dimension of the output latent embedding.
        hidden_dim
            Hidden layer sizes.  Defaults to ``[1024, 256]``.
        dropout
            Dropout rate between hidden layers.
        input_dropout
            Dropout rate applied before the first hidden layer.
        """
        super().__init__()
        hidden_dim = [1024, 256] if hidden_dim is None else hidden_dim
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))
        # self._init_weights()
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier weights and zero bias.

        This helper is kept for experiments that require explicit recursive
        initialization.  The current code path preserves the original CONNECT
        initialization behavior for reproducibility.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Encode an input batch.

        Parameters
        ----------
        x
            Dense tensor of input features.

        Returns
        -------
        torch.Tensor
            L2-normalized latent embedding.
        """
        for layer in self.network:
            x = layer(x)
        return F.normalize(x, p=2, dim=1)


class Decoder(nn.Module):
    """Multilayer perceptron decoder for reconstruction or cross-modal prediction."""

    def __init__(self, n_genes: int, latent_dim: int = 32, hidden_dim: List[int] = None, dropout: float = 0.2, relu: bool = False):
        """Create the decoder network.

        Parameters
        ----------
        n_genes
            Number of output features.
        latent_dim
            Dimension of the input latent embedding.
        hidden_dim
            Hidden layer sizes.  Defaults to ``[1024]`` when omitted.
        dropout
            Dropout rate between hidden layers.
        relu
            Whether to append a ReLU activation to the output layer.
        """
        super().__init__()
        hidden_dim = [1024] if hidden_dim is None else hidden_dim
        self.network = nn.ModuleList()
        self.relu = relu
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(nn.Linear(latent_dim, hidden_dim[i]), nn.BatchNorm1d(hidden_dim[i]), nn.PReLU())
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        in_dim = hidden_dim[-1] if hidden_dim else latent_dim
        self.network.append(nn.Linear(in_dim, n_genes))
        if self.relu:
            self.network.append(nn.ReLU())
        # self._init_weights()
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier weights and zero bias.

        This helper is kept for experiments that require explicit recursive
        initialization.  The current code path preserves the original CONNECT
        initialization behavior for reproducibility.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Decode latent vectors.

        Parameters
        ----------
        x
            Latent tensor with shape ``(n_cells, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed or predicted feature matrix.
        """
        for layer in self.network:
            x = layer(x)
        return x


class Mapping(nn.Module):
    """Linear mapping between the two modality latent spaces."""

    def __init__(self, origin_module_dim: int, target_module_dim: int):
        """Create the latent mapping layer.

        Parameters
        ----------
        origin_module_dim
            Dimension of source latent vectors.
        target_module_dim
            Dimension of target latent vectors.
        """
        super().__init__()
        self.mapping = nn.Linear(origin_module_dim, target_module_dim, bias=True)
        nn.init.xavier_normal_(self.mapping.weight)
        nn.init.zeros_(self.mapping.bias)

    def forward(self, x):
        """Map latent vectors into the paired latent space.

        Parameters
        ----------
        x
            Source latent tensor.

        Returns
        -------
        torch.Tensor
            Latent tensor projected to the target modality space.
        """
        return self.mapping(x)
