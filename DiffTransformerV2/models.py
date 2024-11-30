import torch
import pytorch_lightning as pl
import selfattnmodel
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1Score
from CELoss import MyCrossEntropyLoss
# from utils import plot_confusion_matrix, plot_roc_curve


class Classifier(pl.LightningModule):
    def __init__(self, model, num_classes, lr=None, scheduler=None, config=None):
        super().__init__()
        self.model = model

        kwargs = {"task": "multiclass", "num_classes": num_classes, "average": "macro"}
        self.train_metrics = {
            # "train_acc": Accuracy(**kwargs),
            # "train_auroc": AUROC(**kwargs),
            # "train_precision": Precision(**kwargs),
            # "train_recall": Recall(**kwargs),
            # "train_f1": F1Score(**kwargs),
        }
        self.val_metrics = {
            f"val{name[5:]}": metric.clone()
            for name, metric in self.train_metrics.items()
        }
        self.test_metrics = {
            f"test{name[5:]}": metric.clone()
            for name, metric in self.train_metrics.items()
        }

        self.predictions = []
        self.labels = []

        self.lr = lr
        self.scheduler = scheduler
        self.config = config

        self.loss_fn = MyCrossEntropyLoss()

    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=1)

    def update_metrics(self, metrics, y_hat, y):
        for metric in metrics.values():
            if metric.device != self.device:
                metric.to(self.device)
            metric.update(y_hat, y)

    def log_metrics(self, metrics, **kwargs):
        for name, metric in metrics.items():
            self.log(name, metric.compute(), **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler(optimizer, **self.config),
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        self.update_metrics(self.train_metrics, y_hat, y)
        return loss

    def on_train_epoch_end(self):
        self.log_metrics(self.train_metrics)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.update_metrics(self.val_metrics, y_hat, y)
        self.predictions.append(y_hat[:, 1])
        self.labels.append(y)
        return loss

    def on_validation_epoch_end(self):
        self.log_metrics(self.val_metrics)
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
        )

        labels = torch.cat(self.labels)
        predictions = torch.cat(self.predictions)
        # self.logger.experiment.add_image(
        #     "Confusion Matrix",
        #     plot_confusion_matrix(labels=labels, predictions=predictions),
        #     global_step=self.current_epoch,
        #     dataformats="HWC",
        # )
        # self.logger.experiment.add_image(
        #     "ROC Curve",
        #     plot_roc_curve(labels=labels, predictions=predictions),
        #     global_step=self.current_epoch,
        #     dataformats="HWC",
        # )
        self.predictions.clear()
        self.labels.clear()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.update_metrics(self.test_metrics, y_hat, y)
        return loss

    def on_test_epoch_end(self):
        self.log_metrics(self.test_metrics)


class MILSelfAttentionOne(Classifier):
    def __init__(self, num_classes, args, **kwargs):
        model = selfattnmodel.MILSelfAttention(*args)
        super().__init__(model, num_classes, **kwargs)


###make all your changes right here
class MILSelfAttentionTwo(Classifier):
    def __init__(self, num_classes, args, **kwargs):
        model = selfattnmodel.MILSelfAttention(*args)
        # model = selfattention_single.MILSelfAttention(*args)
        super().__init__(model, num_classes, **kwargs)
