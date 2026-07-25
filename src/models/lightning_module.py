import torch
import pytorch_lightning as pl
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

def build_model():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.05,
    )

class LungTumorSegmentation(pl.LightningModule):
    def __init__(self, lr=5e-4, patch_size=(64, 64, 64)):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model()
        self.loss_fn = DiceCELoss(sigmoid=True, lambda_dice=0.8, lambda_ce=0.2)
        self.metrics = DiceMetric(include_background=True, reduction="mean")
        self.patch_size = patch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mri = batch["mri"]["data"]
        mask = batch["mask"]["data"].float()
        loss = self.loss_fn(self.model(mri), mask)
        self.log("train_loss", loss, prog_bar=True, batch_size=mri.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        mri = batch["mri"]["data"]
        mask = batch["mask"]["data"].float()
        logits = sliding_window_inference(
            inputs=mri, roi_size=self.patch_size, sw_batch_size=2,
            predictor=self.forward, overlap=0.75, mode="gaussian")
        loss = self.loss_fn(logits, mask)
        pred = (torch.sigmoid(logits) > 0.5).float()
        self.metrics(pred, mask)
        tp = (pred * mask).sum()
        self.log("val_loss", loss, prog_bar=True, batch_size=1)
        self.log("val_recall", tp / (mask.sum() + 1e-8), batch_size=1)
        self.log("val_prec", tp / (pred.sum() + 1e-8), prog_bar=True, batch_size=1)
        return loss

    def on_validation_epoch_end(self):
        dice = self.metrics.aggregate().item()
        self.metrics.reset()
        self.log("val_dice", dice, prog_bar=True)
        print(f"Val dice: {dice:.4f}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_dice",
                "interval": "epoch",
                "frequency": 1
            }
        }
