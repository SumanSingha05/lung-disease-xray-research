import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224

# Dataset paths
DATASETS = {
    "chest_xray": {
        "train": "raw_datasets/chest_xray/train",
        "test": "raw_datasets/chest_xray/test"
    },

    "covid": {
        "train": "raw_datasets/covid_19/train",
        "test": "raw_datasets/covid_19/test"
    },

    "nih": {
        "train": "raw_datasets/nih/train",
        "test": "raw_datasets/nih/test",
        "csv": "raw_datasets/nih/Data_Entry_2017.csv"
    }
}