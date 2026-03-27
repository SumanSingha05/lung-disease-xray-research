import torch
from datasets.dataloader import get_dataloader
from models.resnet import get_resnet
from training.train import train_model
from evaluation.evaluate import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

# TRAIN: Kaggle Pneumonia
train_loader, classes = get_dataloader("datasets/kaggle_pneumonia")

model = get_resnet(len(classes))
model = train_model(model, train_loader, device)

# TEST: NIH
test_loader, _ = get_dataloader("datasets/nih")
evaluate(model, test_loader, device)