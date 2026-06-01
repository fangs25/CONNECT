from abc import abstractmethod
from numpy import inf
import numpy as np
import torch
from pathlib import Path
from .util import inf_loop, MetricTracker
import logging
import torch.nn.functional as F

class BaseTrainer:
    """Common training loop with optional early stopping and checkpointing."""

    def __init__(
        self,
        model,
        optimizer,
        epochs=80,
        save_period=0,
        monitor='min val_loss',
        early_stop=3,
        logger=None,
        checkpoint_dir=None,
    ):
        """Initialize shared trainer state, monitoring, and checkpoint options.

        Parameters
        ----------
        model
            PyTorch model to optimize.
        optimizer
            Optimizer constructed from trainable model parameters.
        epochs
            Maximum number of training epochs.
        save_period
            Save a checkpoint every ``save_period`` epochs.  Use ``0`` to
            disable periodic checkpointing.
        monitor
            Early-stopping target in the form ``"min val_loss"``,
            ``"max metric_name"``, or ``"off"``.
        early_stop
            Number of epochs without monitored improvement before stopping.
        logger
            Optional logger.  If omitted, a basic ``train`` logger is created.
        checkpoint_dir
            Directory used for checkpoints.  Checkpoints are written only when
            this path is provided and ``save_period`` is non-zero.
        """
        if logger is None:
            self.logger = logging.getLogger('train')
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.model = model
        self.optimizer = optimizer

        self.epochs = epochs
        self.save_period = save_period
        self.monitor = monitor
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = early_stop
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        # TensorBoard or external writers can be attached here without changing
        # the trainer API used in the experiments.
        self.writer = None

    @abstractmethod
    def _train_epoch(self, epoch):
        """Train or validate one epoch in subclasses.

        Parameters
        ----------
        epoch
            One-based epoch number.

        Returns
        -------
        dict
            Dictionary of scalar metrics for the epoch.
        """
        raise NotImplementedError

    def train(self):
        """Run the full training loop.

        The method calls :meth:`_train_epoch` for each epoch, logs returned
        metrics, applies optional early stopping, and writes checkpoints when
        configured.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # Optional early stopping based on a logged training quantity.
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best - 0.005) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best + 0.005)
                except KeyError:
                    self.logger.warning("Warning: monitor target '{}' is not found. "
                                        "Early stopping is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.logger.info("Metric improved...")
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1


                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if self.save_period and self.checkpoint_dir is not None and epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """Save model and optimizer state.

        Parameters
        ----------
        epoch
            Epoch number stored in the checkpoint filename and state.
        save_best
            If ``True``, also write ``model_best.pth`` in ``checkpoint_dir``.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            # 'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """Resume model and optimizer state from a checkpoint.

        Parameters
        ----------
        resume_path
            Path to a checkpoint produced by :meth:`_save_checkpoint`.
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        # else:
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    """Trainer for the standard paired-data training stage."""
    def __init__(self, model, optimizer, data_loader, epochs=80,
                 valid_data_loader=None, lr_scheduler=None, device = 'cuda:0',
                len_epoch=None, logger = None,):
        """Create the standard paired-data trainer.

        Parameters
        ----------
        model
            :class:`connect.model.MultiModalityAE` instance.
        optimizer
            Optimizer for all trainable model parameters.
        data_loader
            Paired :class:`connect.dataloader.MultiomicsDataLoader` used for
            training.
        epochs
            Number of standard-training epochs.
        valid_data_loader
            Optional validation loader returned by ``data_loader.split_validation``.
        lr_scheduler
            Optional learning-rate scheduler stepped once per epoch.
        device
            Torch device string or object used for tensor transfer.
        len_epoch
            Optional number of iterations per epoch.  When provided, the
            training loader is cycled indefinitely.
        logger
            Optional logger for progress and metric messages.
        """
        super().__init__(model = model, optimizer = optimizer, epochs=epochs, logger = logger)
        self.device = device
        self.data_loader = data_loader


        self.mod1_type = data_loader.dataset.modality_1_type
        self.mod2_type = data_loader.dataset.modality_2_type

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
            if self.len_epoch * self.data_loader.batch_size > 30000:
                self.early_stop = 1
                self.logger.info('For large-scale dataset, set early stop epoch to 1...')
        else:
            # Iteration-based training reuses the loader indefinitely.
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker('loss',
                                        'modality1_map_loss', 'modality2_map_loss',
                                        'inter_CL_loss',
                                        'modality1_recon_loss', 'modality2_recon_loss',
                                        'modality1_pred_loss', 'modality2_pred_loss',
                                        writer=self.writer)
        self.valid_metrics = MetricTracker('loss',
                                        'modality1_map_loss', 'modality2_map_loss',
                                        'inter_CL_loss',
                                        'modality1_recon_loss', 'modality2_recon_loss',
                                        'modality1_pred_loss', 'modality2_pred_loss',
                                        writer=self.writer)


    def _train_epoch(self, epoch):
        """Train one standard paired-data epoch.

        Parameters
        ----------
        epoch
            One-based epoch number.

        Returns
        -------
        dict
            Average training metrics for the epoch, optionally prefixed
            validation metrics when a validation loader is configured.
        """
        self.logger.info('Model Training...')
        self.model.train()
        self.train_metrics.reset()
        # torch.manual_seed(42)

        for batch_idx, (mod1_arr, mod2_arr, mod1_arr_raw, mod2_arr_raw) in enumerate(self.data_loader):


            if batch_idx == 0 and epoch == 1:
                self.logger.info('Convering batch data to GPU...')

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.forward(mod1_arr.to(self.device), mod2_arr.to(self.device))

                for x in outputs:
                    print(x.mean().item(), x.std().item(), x.abs().sum().item())
                print(mod1_arr.sum(), mod2_arr.sum(), mod1_arr.max(), mod2_arr.max())
                self.model.train()
                
            # self.logger.info("%s MB allocated | %s MB reserved", torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2)
            mod1_arr, mod2_arr = mod1_arr.to(self.device), mod2_arr.to(self.device)
            mod1_arr_raw, mod2_arr_raw = mod1_arr_raw.to(self.device), mod2_arr_raw.to(self.device)

            self.optimizer.zero_grad()

            if batch_idx == 0 and epoch == 1:
                self.logger.info('Forwarding data...')

            mod1_latent, mod2_latent, \
            mod1_recon_from_mod1, mod2_recon_from_mod2, \
            mod2_predicted_from_mod1, mod1_predicted_from_mod2, \
            mod2_mapping_from_mod1, mod1_mapping_from_mod2 = \
            self.model.forward(mod1_arr, mod2_arr)


            if batch_idx == 0 and epoch == 1:
                self.logger.info('Computing loss...')

            modality1_recon_loss = self.model.ae_modality_1.recon_mse(mod1_arr_raw, mod1_recon_from_mod1) # rercon
            modality2_pred_loss = self.model.ae_modality_2.pred_mse(mod1_arr_raw, mod1_predicted_from_mod2)

            if self.data_loader.dataset.modality_2_type == 'ATAC':
                modality1_pred_loss = self.model.ae_modality_1.pred_mse(mod2_arr_raw, mod2_predicted_from_mod1)
                modality2_recon_loss = self.model.ae_modality_2.recon_mse(mod2_arr_raw, mod2_recon_from_mod2)
            else:
                modality1_pred_loss = self.model.ae_modality_1.pred_mse(mod2_arr, mod2_predicted_from_mod1)
                modality2_recon_loss = self.model.ae_modality_2.recon_mse(mod2_arr, mod2_recon_from_mod2)

            modality1_map_loss = self.model.ae_modality_1.mapping_loss(mod2_latent, mod2_mapping_from_mod1) # mapping
            modality2_map_loss = self.model.ae_modality_2.mapping_loss(mod1_latent, mod1_mapping_from_mod2)

            inter_CL_loss = self.model.contrastive_loss(mod1_latent, mod2_latent)

            recon_loss = modality1_recon_loss + modality2_recon_loss
            cl_loss = inter_CL_loss
            map_loss = modality1_map_loss + modality2_map_loss
            pred_loss = modality1_pred_loss + modality2_pred_loss
            
            w_map, w_cl, w_recon, w_pred = self.model.weights[0], self.model.weights[1], self.model.weights[2],self.model.weights[3]

            total_loss = (
                w_recon * recon_loss +
                w_cl * cl_loss +
                w_map * map_loss +
                w_pred * pred_loss
            )
            nativa_loss = recon_loss + cl_loss + map_loss + pred_loss


            if batch_idx == 0 and epoch == 1:
                self.logger.info('Backwarding loss...')
                self.logger.info(f'{w_map}, {w_cl}, {w_recon}, {w_pred}')

            total_loss.backward()
            self.optimizer.step()


            self.train_metrics.update('loss', nativa_loss.item())

            self.train_metrics.update('inter_CL_loss', inter_CL_loss.item())
            self.train_metrics.update('modality1_recon_loss', modality1_recon_loss.item())
            self.train_metrics.update('modality2_recon_loss', modality2_recon_loss.item())
            self.train_metrics.update('modality1_map_loss', modality1_map_loss.item())
            self.train_metrics.update('modality2_map_loss', modality2_map_loss.item())
            self.train_metrics.update('modality1_pred_loss', modality1_pred_loss.item())
            self.train_metrics.update('modality2_pred_loss', modality2_pred_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_loss.item()))            

            if batch_idx == self.len_epoch:
                break


        log = self.train_metrics.result()
        self.logger.info(f'Do validation{self.do_validation}...')
        if self.do_validation:
            self.logger.info('Performing validation...')
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """Evaluate one validation epoch.

        Parameters
        ----------
        epoch
            One-based epoch number.

        Returns
        -------
        dict
            Average validation losses for reconstruction, prediction, mapping,
            contrastive matching, and their unweighted sum.
        """
        self.model.eval()
        self.valid_metrics.reset()


        with torch.no_grad():
            for batch_idx, (mod1_arr, mod2_arr, mod1_arr_raw, mod2_arr_raw) in enumerate(self.valid_data_loader):


                mod1_arr, mod2_arr = mod1_arr.to(self.device), mod2_arr.to(self.device)
                mod1_arr_raw, mod2_arr_raw = mod1_arr_raw.to(self.device), mod2_arr_raw.to(self.device)

                mod1_latent, mod2_latent, \
                mod1_recon_from_mod1, mod2_recon_from_mod2, \
                mod2_predicted_from_mod1, mod1_predicted_from_mod2, \
                mod2_mapping_from_mod1, mod1_mapping_from_mod2 = \
                self.model.forward(mod1_arr, mod2_arr)

                modality1_recon_loss = self.model.ae_modality_1.recon_mse(mod1_arr_raw, mod1_recon_from_mod1) # rercon
                modality2_pred_loss = self.model.ae_modality_2.pred_mse(mod1_arr_raw, mod1_predicted_from_mod2)

                if self.data_loader.dataset.modality_2_type == 'ATAC':
                    modality1_pred_loss = self.model.ae_modality_1.pred_mse(mod2_arr_raw, mod2_predicted_from_mod1)
                    modality2_recon_loss = self.model.ae_modality_2.recon_mse(mod2_arr_raw, mod2_recon_from_mod2)
                else:
                    modality1_pred_loss = self.model.ae_modality_1.pred_mse(mod2_arr, mod2_predicted_from_mod1)
                    modality2_recon_loss = self.model.ae_modality_2.recon_mse(mod2_arr, mod2_recon_from_mod2)

                modality1_map_loss = self.model.ae_modality_1.mapping_loss(mod2_latent, mod2_mapping_from_mod1) # mapping
                modality2_map_loss = self.model.ae_modality_2.mapping_loss(mod1_latent, mod1_mapping_from_mod2)

                inter_CL_loss = self.model.contrastive_loss(mod1_latent, mod2_latent)
                
                recon_loss = modality1_recon_loss + modality2_recon_loss
                cl_loss = inter_CL_loss
                map_loss = modality1_map_loss + modality2_map_loss
                pred_loss = modality1_pred_loss + modality2_pred_loss# + ranking_loss
                
                native_loss = recon_loss + cl_loss + map_loss + pred_loss

                self.valid_metrics.update('loss', native_loss.item())

                self.valid_metrics.update('modality1_recon_loss', modality1_recon_loss.item())
                self.valid_metrics.update('modality2_recon_loss', modality2_recon_loss.item())
                self.valid_metrics.update('modality1_map_loss', modality1_map_loss.item())
                self.valid_metrics.update('modality2_map_loss', modality2_map_loss.item())
                self.valid_metrics.update('modality1_pred_loss', modality1_pred_loss.item())
                self.valid_metrics.update('modality2_pred_loss', modality2_pred_loss.item())
                self.valid_metrics.update('inter_CL_loss', inter_CL_loss.item())
        


        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """Format progress information for epoch logs.

        Parameters
        ----------
        batch_idx
            Current mini-batch index.

        Returns
        -------
        str
            Progress string containing processed samples, total samples, and
            percentage completion.
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class AugmentTrainer(BaseTrainer):
    """Trainer that interleaves paired training with single-modality augmentation."""

    def __init__(
        self,
        model,
        optimizer,
        data_loader,
        valid_data_loader=None,
        unimodal_loader=None,
        unimodal_optimizer=None,
        unimodal_interval=3,
        lr_scheduler=None,
        device='cuda:0',
        epochs = 10,
        len_epoch=None,
        logger=None,
    ):
        """Create the augmentation-stage trainer.

        Parameters
        ----------
        model
            :class:`connect.model.MultiModalityAE` instance.
        optimizer
            Optimizer used for the paired-data update.
        data_loader
            Paired dataloader used during the augmentation stage.
        valid_data_loader
            Reserved optional validation loader.
        unimodal_loader
            Single-modality dataloader used for periodic augmentation updates.
        unimodal_optimizer
            Optimizer used for the single-modality augmentation update.
        unimodal_interval
            Run one single-modality update every ``unimodal_interval`` paired
            mini-batches.
        lr_scheduler
            Optional learning-rate scheduler stepped once per epoch.
        device
            Torch device used for tensor transfer.
        epochs
            Number of augmentation epochs.
        len_epoch
            Optional number of iterations per epoch.  When provided, the paired
            loader is cycled indefinitely.
        logger
            Optional logger for progress messages.
        """
        super().__init__(model=model, optimizer=optimizer, logger=logger, epochs=epochs)

        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.unimodal_loader = unimodal_loader
        self.unimodal_iter = iter(unimodal_loader) if unimodal_loader else None
        self.unimodal_optimizer = unimodal_optimizer
        self.unimodal_interval = unimodal_interval
        self.lr_scheduler = lr_scheduler
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
            if self.len_epoch * self.data_loader.batch_size > 30000:
                self.early_stop = 1
                self.logger.info('For large-scale dataset, set early stop epoch to 1...')
        else:
            # Iteration-based training reuses the loader indefinitely.
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss',
                                'map_loss', 
                                'inter_CL_loss',
                                'recon_loss', 
                                'pred_loss',
                                writer=self.writer)

    def unimodal_loss(self, x, x_raw, encoder, ae):
        """Combine reconstruction and VICReg-style latent regularization.

        Parameters
        ----------
        x
            Processed single-modality input batch.
        x_raw
            Raw-count target batch for reconstruction.
        encoder
            Branch-forward function, typically ``model.forward_modality1`` or
            ``model.forward_modality2``.
        ae
            Corresponding :class:`connect.model.ContrastiveAE` branch providing
            the reconstruction loss.

        Returns
        -------
        torch.Tensor
            Scalar augmentation loss.
        """
        z, x_recon, _, _ = encoder(x)

        # reconstruction
        recon_loss = ae.recon_mse(x_raw, x_recon)

        # latent variance (VICReg-style)
        var = torch.var(z, dim=0)
        var_loss = torch.mean(F.relu(1.0 - torch.sqrt(var + 1e-4)))

        # covariance
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        cov_loss = (cov.fill_diagonal_(0) ** 2).sum() / z.shape[1]

        return recon_loss + 0.1 * var_loss + 0.01 * cov_loss
    
    def _train_epoch(self, epoch):
        """Train one augmentation epoch with periodic single-modality updates.

        Parameters
        ----------
        epoch
            One-based epoch number.

        Returns
        -------
        dict
            Average augmentation-stage losses for the epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (mod1_arr, mod2_arr, mod1_raw, mod2_raw) in enumerate(self.data_loader):

            if batch_idx == 0 and epoch == 1:
                self.logger.info('Convering batch data to GPU...')

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.forward(mod1_arr.to(self.device), mod2_arr.to(self.device))

                for x in outputs:
                    print(x.mean().item(), x.std().item(), x.abs().sum().item())
                print(mod1_arr.sum(), mod2_arr.sum(), mod1_arr.max(), mod2_arr.max())
                self.model.train()

            # ======== 1. UNIMODAL STEP (periodic) ========
            if self.unimodal_loader and batch_idx % self.unimodal_interval == 0:
                try:
                    uni_x, uni_x_raw = next(self.unimodal_iter)
                except StopIteration:
                    self.unimodal_iter = iter(self.unimodal_loader)
                    uni_x, uni_x_raw = next(self.unimodal_iter)

                uni_x = uni_x.to(self.device)
                uni_x_raw = uni_x_raw.to(self.device)

                self.unimodal_optimizer.zero_grad()

                # only encoder forward
                if self.unimodal_loader.dataset.modality_1_type == 'RNA':
                    loss_uni = self.unimodal_loss(
                        uni_x, uni_x_raw,
                        self.model.forward_modality1,
                        self.model.ae_modality_1
                    )
                else:
                    loss_uni = self.unimodal_loss(
                        uni_x, uni_x_raw,
                        self.model.forward_modality2,
                        self.model.ae_modality_2
                    )

                loss_uni.backward()
                self.unimodal_optimizer.step()

            # ======== 2. PAIRED STEP ========
            mod1_arr = mod1_arr.to(self.device)
            mod2_arr = mod2_arr.to(self.device)
            mod1_raw = mod1_raw.to(self.device)
            mod2_raw = mod2_raw.to(self.device)

            self.optimizer.zero_grad()

            (
                z1, z2,
                x1_rec, x2_rec,
                x2_pred, x1_pred,
                z2_map, z1_map
            ) = self.model.forward(mod1_arr, mod2_arr)

            # losses
            if self.data_loader.dataset.modality_2_type == 'ATAC':
                rec_loss = (
                    self.model.ae_modality_1.recon_mse(mod1_raw, x1_rec) +
                    self.model.ae_modality_2.recon_mse(mod2_raw, x2_rec)
                )
                pred_loss = (
                    self.model.ae_modality_1.pred_mse(mod2_raw, x2_pred) +
                    self.model.ae_modality_2.pred_mse(mod1_raw, x1_pred)
                )
            else:
                rec_loss = (
                    self.model.ae_modality_1.recon_mse(mod1_raw, x1_rec) +
                    self.model.ae_modality_2.recon_mse(mod2_arr, x2_rec)
                )
                pred_loss = (
                    self.model.ae_modality_1.pred_mse(mod2_arr, x2_pred) +
                    self.model.ae_modality_2.pred_mse(mod1_raw, x1_pred)
                )  # ADT 使用 mod2_arr
            


            map_loss = (
                self.model.ae_modality_1.mapping_loss(z2, z2_map) +
                self.model.ae_modality_2.mapping_loss(z1, z1_map)
            )



            cl_loss = self.model.contrastive_loss(z1, z2)

            w_map, w_cl, w_rec, w_pred = self.model.weights

            total_loss = (
                w_rec * rec_loss +
                w_map * map_loss +
                w_pred * pred_loss +
                w_cl * cl_loss
            )
            nativa_loss = rec_loss + cl_loss + map_loss + pred_loss

            total_loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', nativa_loss.item())

            self.train_metrics.update('inter_CL_loss', cl_loss.item())
            self.train_metrics.update('recon_loss', rec_loss.item())
            self.train_metrics.update('map_loss', map_loss.item())
            self.train_metrics.update('pred_loss', pred_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info('Augment Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_loss.item()))            

            if batch_idx == self.len_epoch:
                break


        log = self.train_metrics.result()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log
    
    def _progress(self, batch_idx):
        """Format progress information for augmentation logs.

        Parameters
        ----------
        batch_idx
            Current mini-batch index.

        Returns
        -------
        str
            Progress string for logging.
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class AlignTrainer(BaseTrainer):
    """Trainer for final latent-space mapping alignment."""

    def __init__(
        self,
        model,
        optimizer,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        device='cuda:0',
        epochs = 10,
        len_epoch=None,
        logger=None,
        w_align=1.0,
        w_iso=0.1,
        w_cl=0.05,
    ):
        """Create the alignment-stage trainer and loss weights.

        Parameters
        ----------
        model
            :class:`connect.model.MultiModalityAE` instance.
        optimizer
            Optimizer for alignment-stage trainable parameters.
        data_loader
            Paired dataloader used for alignment.
        valid_data_loader
            Reserved optional validation loader.
        lr_scheduler
            Optional learning-rate scheduler stepped once per epoch.
        device
            Torch device used for tensor transfer.
        epochs
            Number of alignment epochs.
        len_epoch
            Optional number of iterations per epoch.
        logger
            Optional logger for progress messages.
        w_align
            Weight on direct latent alignment loss.
        w_iso
            Weight on isometry regularization.
        w_cl
            Weight on contrastive loss during alignment.
        """
        super().__init__(model=model, optimizer=optimizer, epochs = epochs, logger=logger)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.len_epoch = len_epoch or len(data_loader)
        self.w_align = w_align
        self.w_iso = w_iso
        self.w_cl = w_cl
        self.log_step = int(np.sqrt(data_loader.batch_size))


    def _isometry_loss(self, z_src, z_tgt):
        """Penalize changes in pairwise latent displacement after mapping.

        Parameters
        ----------
        z_src
            Source latent tensor before mapping.
        z_tgt
            Target or mapped latent tensor to compare against ``z_src``.

        Returns
        -------
        torch.Tensor
            Scalar MSE penalty on sampled pairwise displacements.
        """
        idx = torch.randperm(z_src.size(0))[: z_src.size(0) // 2]
        dz_src = z_src[idx] - z_src[idx.flip(0)]
        dz_tgt = z_tgt[idx] - z_tgt[idx.flip(0)]
        return F.mse_loss(dz_src, dz_tgt)

    def _train_epoch(self, epoch):
        """Train one alignment epoch.

        Parameters
        ----------
        epoch
            One-based epoch number.

        Returns
        -------
        dict
            Average alignment loss for the epoch.
        """
        self.model.train()
        total_loss_epoch = 0.0

        for batch_idx, (mod1_arr, mod2_arr, _, _) in enumerate(self.data_loader):

            if batch_idx == 0 and epoch == 1:
                self.logger.info('Convering batch data to GPU...')

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.forward(mod1_arr.to(self.device), mod2_arr.to(self.device))

                for x in outputs:
                    print(x.mean().item(), x.std().item(), x.abs().sum().item())
                print(mod1_arr.sum(), mod2_arr.sum(), mod1_arr.max(), mod2_arr.max())
                self.model.train()

            mod1_arr = mod1_arr.to(self.device)
            mod2_arr = mod2_arr.to(self.device)

            self.optimizer.zero_grad()

            (
                z1, z2,
                _, _, _, _,
                z2_map, z1_map
            ) = self.model.forward(mod1_arr, mod2_arr)

            # 1. alignment loss
            align_loss = (
                F.mse_loss(z1, z1_map) +
                F.mse_loss(z2, z2_map)
            )

            # 2. isometry loss
            iso_loss = (
                self._isometry_loss(z1, z1_map) +
                self._isometry_loss(z2, z2_map)
            )

            # 3. contrastive (optional)
            cl_loss = self.model.contrastive_loss(z1, z2)

            loss = (
                self.w_align * align_loss +
                self.w_iso * iso_loss +
                self.w_cl * cl_loss
            )

            loss.backward()
            self.optimizer.step()

            total_loss_epoch += loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.info('Align Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))    

            if batch_idx == self.len_epoch:
                break

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return {"align_loss": total_loss_epoch / self.len_epoch}

    def _progress(self, batch_idx):
        """Format progress information for alignment logs.

        Parameters
        ----------
        batch_idx
            Current mini-batch index.

        Returns
        -------
        str
            Progress string for logging.
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
