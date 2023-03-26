import ttach as tta
import pytorch_lightning as pl
from torchmetrics import F1Score

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch_optimizer as extra_optim
from sam import SAM

import timm
import segmentation_models_pytorch as smp

from typing import List, Any, Type
from decoders.MANet import MANet
from segmentation_models_pytorch.base import SegmentationHead
from decoders.fapn import FaPNHead


class FAPN(nn.Module):
    def __init__(self, encoder_name='resnet34', in_chans=4 * 6,
                 num_classes=1,
                 pretrained=True,
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.encoder = timm.create_model(encoder_name,
                                         in_chans=in_chans,
                                         pretrained=pretrained,
                                         features_only=True)

        feature_channels = [in_chans, *self.encoder.feature_info.channels()]

        encoder_depth = len(self.encoder.feature_info.channels())
        decoder_channels = [16 * (2 ** i) for i in range(encoder_depth)][::-1]

        self.decoder = FaPNHead(
            in_channels=feature_channels,
            channel=128,
        )
        self.decoder.conv_seg = nn.Identity()  # to use uniform segmentation head
        decoder_channels[-1] = 128  # to use uniform segmentation head

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=None,  # activation,
            kernel_size=3,
            upsampling=2,  # if encoder_name not in smp.encoders.get_encoder_names() else 1,
        )

        self.modify_head()

    def modify_head(self):
        with torch.no_grad():
            x = torch.rand(2, self.in_chans, 256, 256)
            y = self.forward(x)

        div = y.shape[-1] // x.shape[-1]
        if div != 1:
            for m in self.segmentation_head.modules():
                if hasattr(m, 'scale_factor'):
                    m.scale_factor = m.scale_factor // div

    def forward(self, x):
        # x - b (c t) h w

        features = [x, *self.encoder(x)]
        decoder_output = self.decoder(*features)

        out = self.segmentation_head(decoder_output)

        return out


class BoundaryModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        if not hasattr(self.args, 'reduce_months'):
            self.args.reduce_months = False

        num_classes = 1 + (1 if args.include_extent else 0) + (1 if args.include_distance else 0)
        num_bands = {True: 3, False: 4}[self.args.use_visible_bands_only]
        in_chans = num_bands * 6
        self.in_chans = in_chans

        if args.model in smp.encoders.get_encoder_names():
            encoder_name = args.model
        else:
            print('Using Timm encoder')
            encoder_name = f"tu-{args.model}"

        if args.decoder == 'unet':
            self.net = smp.Unet(
                encoder_name=encoder_name,  # 'efficientnet-b3',
                encoder_weights='imagenet',
                in_channels=in_chans,  # 3,
                classes=num_classes,
            )
        elif args.decoder == 'unet++':
            self.net = smp.UnetPlusPlus(
                encoder_name=encoder_name,  # 'efficientnet-b3',
                encoder_weights='imagenet',
                in_channels=in_chans,  # 3,
                classes=num_classes,
            )
        elif args.decoder == 'manet':
            self.net = smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                decoder_pab_channels=64,
                in_channels=in_chans,
                classes=num_classes,
            )
        elif args.decoder == 'MANet':  # completely different MANet
            self.net = MANet(
                backbone_name=args.model,
                num_channels=in_chans,
                num_classes=num_classes,
                pretrained=True,
            )
        elif args.decoder == 'deeplabv3':
            self.net = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                decoder_channels=128,
                encoder_depth=4,
                upsampling=4,
                in_channels=in_chans,
                classes=num_classes,
            )
        elif args.decoder == 'deeplabv3+':
            self.net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                decoder_channels=128,
                encoder_depth=4,
                upsampling=4,
                in_channels=in_chans,
                classes=num_classes,
            )
        elif args.decoder == 'fapn':
            self.net = FAPN(
                encoder_name=args.model,
                in_chans=in_chans,
                num_classes=num_classes,
            )

        self.loss_boundary = smp.losses.DiceLoss(
            mode='binary', classes=None,
#             log_loss=args.log_loss,
            from_logits=True,
            smooth=0.01,
        )
        self.loss_extent = smp.losses.DiceLoss(
                mode='binary', classes=None,
#                 log_loss=args.log_loss,
                from_logits=True,
                smooth=0.01,
            )
        self.loss_distance = nn.MSELoss()

        self._metric = F1Score(
            task='binary',
            threshold=0.5
        )
        self.num_classes = num_classes
        if self.args.optimizer == 'sam':
            self.automatic_optimization = False

        self.modify_head()

    def modify_head(self):
        with torch.no_grad():
            x = torch.rand(2, self.in_chans, 256, 256)
            y = self.forward(x)

        div = y.shape[-1] / x.shape[-1]
        if div != 1:
            for m in self.net.segmentation_head.modules():
                if hasattr(m, 'scale_factor'):
                    m.scale_factor = int(m.scale_factor / div)

    def forward(self, x):
        x = self.net(x)
        return x

    def _compute_metric(self, output, target_var, mask_var=None):
        with torch.no_grad():
            metric = self._metric(output[:, 0], target_var[:, 0].long())
        return metric

    def _compute_loss(self, output, target_var):
        loss1 = self.loss_boundary(output[:, 0], target_var[:, 0])  # Boundary
        loss2 = 0
        if self.args.include_extent:
            loss2 = self.loss_extent(output[:, 1], target_var[:, 1])  # extent
        loss3 = 0
        if self.args.include_distance:
            loss3 = self.loss_distance(output[:, 2].sigmoid(), target_var[:, 2])  # distance
        loss = loss1 + loss2 + loss3

        return loss

    def training_step(self, batch, batch_nb):
        input_var, target_var = batch['image'], batch['mask']
        batch_size = input_var.shape[0]

        output = self(input_var)
        loss = self._compute_loss(output, target_var)  # , mask_var)

        def closure():
            output = self(input_var)
            loss = self._compute_loss(output, target_var)
            loss.backward()
            return loss

        if self.args.optimizer == 'sam':
            optimizer = self.optimizers()
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()

            # single scheduler
            sch = self.lr_schedulers()
            if sch is not None:
                #                 sch.step()
                # step every N epochs
                N = 1
                if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % N == 0:
                    sch.step()

        f1 = self._compute_metric(output, target_var)  # , mask_var)

        self.log('f1', f1, logger=True, prog_bar=True, batch_size=batch_size)
        self.log('train_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train_f1', f1, logger=True, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        optimizer = self.optimizers()
        self.log('lr', optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss

    #     def validation_step(self, batch, batch_idx):
    #         batch = self.process_batch(batch)
    #         input_var, target_var = batch

    #         output = self(input_var)
    #         loss = self._compute_loss(output, target_var)  #, mask_var)
    #         f1 = self._compute_metric(output, target_var)  #, mask_var)

    #         self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    #         self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        other_args = {}
        if self.args.optimizer == 'sgd':
            other_args = {'lr': self.args.lr, 'momentum': 0.9,
                          'weight_decay': self.args.weight_decay,
                          'nesterov': True}
        elif self.args.optimizer in ['adam', 'radam']:
            other_args = {'lr': self.args.lr, 'eps': 1e-8,
                          'betas': (0.9, 0.999),
                          'weight_decay': self.args.weight_decay}
        elif self.args.optimizer.startswith('ada'):  # == 'adamw':
            other_args = {'lr': self.args.lr,
                          'weight_decay': self.args.weight_decay}
        elif self.args.optimizer == 'lamb':
            other_args = {'lr': self.args.lr, 'eps': 1e-8,
                          'betas': (0.9, 0.999),
                          'clamp_value': 10,
                          'weight_decay': self.args.weight_decay}
        elif self.args.optimizer in ['sam', 'samsgd']:
            other_args = {'lr': self.args.lr,  # 'momentum': 0.9,
                          'weight_decay': self.args.weight_decay,
                          # 'nesterov': True,
                          'adaptive': True,
                          'base_optimizer': optim.AdamW,  # optim.SGD,
                          'rho': 2}
        optimizer_dict = {'adam': optim.Adam,
                          'sgd': optim.SGD,
                          'radam': optim.RAdam,
                          'lamb': extra_optim.Lamb,
                          'adamw': optim.AdamW,
                          'adabelief': extra_optim.AdaBelief,
                          'sam': SAM,
                          }
        optimizer = optimizer_dict[self.args.optimizer](self.parameters(), **other_args)
        # n_iter_per_epoch = math.ceil(self.args.train_dims /
        #                              (self.args.batch_size * len(self.args.gpus.split(','))))  # * self.args.nodes))
        # num_steps = int(self.args.epochs * n_iter_per_epoch)
        # warmup_steps = int(self.args.warmup * n_iter_per_epoch)

        if self.args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.epochs // 10,  # 20,
                                                        gamma=0.5)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        elif self.args.scheduler == 'onecycle':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.args.lr,
                    total_steps=self.trainer.estimated_stepping_batches,
                ),
                'interval': 'step'
            }
        else:  # None
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class EnsembleVotingModel(pl.LightningModule):
    def __init__(
            self,
            model_cls: Type[pl.LightningModule],
            checkpoint_paths: List[str],
            use_tta=False,
            tta_mode='mean',
            threshold=0.5,
            soft=True,  # decide if using simple averaging (True) or mode based (False)
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.soft = soft
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p, strict=False) for p in checkpoint_paths])

        self.models = nn.ModuleList(
            [nn.Sequential(model, nn.Sigmoid()) for model in self.models]
        )

        if use_tta:
            # defined 2 * 2 * 2 = 8 augmentations !
            transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 90]),
                ]
            )
            self.models = torch.nn.ModuleList(
                [
                    tta.SegmentationTTAWrapper(
                        model,
                        transforms,
                        merge_mode=tta_mode,
                    ) for model in self.models]
            )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        input_var = batch['image']
        logits = torch.stack([m(input_var) for m in self.models])
        
        if self.soft:
            preds = 1 * (logits.mean(0) > self.threshold)
        else:
            num_models = len(logits)
            preds = (logits > self.threshold).sum(axis=1)
            preds = 1 * (preds >= ((num_models + 1) // 2))
        return preds
