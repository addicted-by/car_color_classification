import os
from typing import Any

import lightning.pytorch as pl
import omegaconf
import torch
import torchvision
from natsort import natsorted


class CustomModel(pl.LightningModule):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_name = self.cfg.model.model_name
        self.n_classes = self.cfg.data.n_classes
        self.model = CustomModel.get_model_by_name(
            self.model_name, self.n_classes, self.cfg
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    @staticmethod
    def get_model_by_name(
        model_name: str, n_classes: int, cfg: omegaconf.dictconfig.DictConfig
    ):
        pretrained = cfg.model.pretrained
        if model_name == "resnet101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif model_name == "resnet50":
            model = torchvision.models.resnet50(pretrained=pretrained)

        else:
            raise ValueError(f"Cannot process this model name: {model_name}!")

        model.fc = torch.nn.Linear(2048, n_classes)
        if cfg.model.get("softmax"):
            model.add_module("SoftMax", torch.nn.Softmax(dim=-1))

        if cfg.model.get("ckpt_load"):
            ckpt = cfg.model.ckpt_load
            if ckpt == "last":
                ckpts_path = os.path.join(cfg.artifacts.dirpath, cfg.model.model_name)
                last_ckpt = natsorted(os.listdir(ckpts_path))[-1]
                ckpt = os.path.join(ckpts_path, last_ckpt)

            print(f"Loading checkpoint {ckpt}")
            err_msg = f"Checkpoint {ckpt} does not exists. Please check!"
            assert os.path.exists(ckpt), err_msg
            model.load_state_dict(torch.load(ckpt))

        for name, parameters in model.named_parameters():
            if any(layer in name for layer in cfg.model.freeze_layers):
                parameters.requires_grad_(False)
                print(name, parameters.shape, "Freezed", sep=" |\t", end=" |\n")
        print(model)
        return model

    def forward(self, data):
        return self.model(data)

    def common_step(self, batch: Any, batch_idx):
        data, target = batch
        y_preds = self(data)
        loss = self.loss_fn(y_preds, target)
        predictions = y_preds.argmax(dim=1)
        correct = (predictions == target).sum().item()
        accuracy = correct / data.shape[0]

        return loss, accuracy

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass

    def configure_optimizers(self) -> Any:
        # param_optimizer = list(self.named_parameters())
        # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": self.cfg.trainer.weight_decay,
        #     },
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.0,
        #     },
        # ]

        # optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters,
        #     **self.cfg.trainer.opt_cfg,
        # )

        optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=5e-5,
        )

        if self.cfg.trainer.get("lr_scheduler"):
            if self.cfg.trainer.lr_scheduler == "cyclic":
                lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer, **self.cfg.trainer.lr_scheduler_cfg
                )
            elif self.cfg.trainer.lr_scheduler == "step":
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, **self.cfg.trainer.lr_scheduler_cfg
                )
            else:
                raise NotImplementedError("Scheduler is not implemented yet.")
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

        else:
            lr_scheduler = None
            return [optimizer]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
