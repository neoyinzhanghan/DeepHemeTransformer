import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import numpy as np
import pytorch_lightning as pl
import os
import io
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
)
from matplotlib import pyplot as plt


# Returns an image of the confusion matrix created from the labels and predictions
def plot_confusion_matrix(labels, predictions):
    fig, ax = plt.subplots(figsize=(2, 2))
    cm = confusion_matrix(
        labels.cpu(), (predictions >= 0.5).cpu().int(), normalize="true"
    )
    disp = ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Reds")
    disp.im_.set_clim(0, 1)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="raw")
    buf.seek(0)
    im = np.reshape(
        np.frombuffer(buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    plt.close(fig)
    buf.close()
    return im


# Returns an image of the roc curve created from the labels and predictions
def plot_roc_curve(labels, predictions):
    fig, ax = plt.subplots(figsize=(2, 2))
    fpr, tpr, _ = roc_curve(labels.cpu(), predictions.cpu())
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=0).plot(
        ax=ax, color="cornflowerblue", lw=1.5
    )
    ax.get_legend().remove()
    ax.plot([0, 1], [0, 1], "--", color="darkblue", lw=1.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="raw")
    buf.seek(0)
    im = np.reshape(
        np.frombuffer(buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    plt.close(fig)
    buf.close()
    return im


def create_train_test_val(patient_data_dir, proportions, tabular_data=[]):

    # Read in data plus inputted proportions
    train_prop, test_prop, val_prop = proportions
    patient_labels = pd.read_csv(
        os.path.join(patient_data_dir, "patient_labels.csv"), dtype={"MRN": str}
    )[["MRN", "SLIDE_ID", "LABEL"] + tabular_data]

    # Get a new dataframe of the unique MRNs
    patient_outcomes = []
    for mrn in patient_labels["MRN"].unique():
        patient_outcomes.append(
            [mrn, patient_labels[patient_labels["MRN"] == mrn]["LABEL"].iloc[0]]
        )
    patient_outcomes = pd.DataFrame(patient_outcomes, columns=["MRN", "LABEL"])

    # Make a split on train and test with no shared patients
    train_df, test_df = train_test_split(
        patient_outcomes, test_size=test_prop, stratify=patient_outcomes["LABEL"]
    )
    train_df = train_df.drop("LABEL", axis=1).merge(
        patient_labels, how="inner", on="MRN"
    )
    test_df = test_df.drop("LABEL", axis=1).merge(patient_labels, how="inner", on="MRN")

    # Make a split on train and val allowing for shared patients
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_prop / (train_prop + val_prop),
        stratify=train_df["LABEL"],
    )

    # Save the files
    train_df.to_csv(os.path.join(patient_data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(patient_data_dir, "test.csv"), index=False)
    val_df.to_csv(os.path.join(patient_data_dir, "val.csv"), index=False)


class TensorDataset(Dataset):
    def __init__(self, patient_data_file, image_data_dir, T_rbc, T_wbc):
        super().__init__()

        if (T_rbc < 0) or (T_wbc < 0):
            raise ValueError("Negative values not allowed")
        if (T_rbc == 0) and (T_wbc == 0):
            raise ValueError("Both values cannot be zero")

        self.T_wbc = T_wbc
        self.T_rbc = T_rbc
        self.wbc_paths = []
        self.rbc_paths = []
        patient_data = pd.read_csv(patient_data_file, dtype={"MRN": str})

        for path in patient_data["MRN"] + "_" + patient_data["SLIDE_ID"].astype(str):
            if self.T_rbc > 0:
                self.rbc_paths.append(
                    os.path.join(
                        image_data_dir, f"{path}/rbc/features/features_original.pt"
                    )
                )
            if self.T_wbc > 0:
                self.wbc_paths.append(
                    os.path.join(
                        image_data_dir, f"{path}/wbc/features/features_original.pt"
                    )
                )
        self.labels = patient_data["LABEL"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.T_rbc > 0:
            rbc = torch.load(self.rbc_paths[idx], weights_only=False)
            rbc = torch.nn.functional.pad(
                rbc, (0, 0, 0, max(0, self.T_rbc - rbc.shape[0]))
            )
            rbc_indices = np.random.choice(
                np.arange(self.T_rbc), self.T_rbc, replace=False
            )
            rbc = rbc[rbc_indices]

            if self.T_wbc == 0:
                return rbc, label, rbc_indices

        if self.T_wbc > 0:
            wbc = torch.load(self.wbc_paths[idx], weights_only=False)
            wbc = torch.nn.functional.pad(
                wbc, (0, 0, 0, max(0, self.T_wbc - wbc.shape[0]))
            )
            wbc_indices = np.random.choice(
                np.arange(self.T_wbc), self.T_wbc, replace=False
            )
            wbc = wbc[wbc_indices]

            if self.T_rbc == 0:
                return wbc, label, wbc_indices

        return (rbc, wbc), label, (rbc_indices, wbc_indices)


class TensorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        patient_data_dir,
        image_data_dir,
        batch_size,
        num_workers,
        T_rbc=121,
        T_wbc=114,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = TensorDataset(
            os.path.join(patient_data_dir, "train.csv"), image_data_dir, T_rbc, T_wbc
        )
        self.val_dataset = TensorDataset(
            os.path.join(patient_data_dir, "val.csv"), image_data_dir, T_rbc, T_wbc
        )
        self.test_dataset = TensorDataset(
            os.path.join(patient_data_dir, "test.csv"), image_data_dir, T_rbc, T_wbc
        )

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
