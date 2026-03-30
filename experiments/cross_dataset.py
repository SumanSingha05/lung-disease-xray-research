from config import DATASETS
from datasets.dataset_loader import load_kaggle, load_covid, load_nih
from models.resnet_model import get_model
from training.trainer import train
from training.losses import get_loss
from evaluation.evaluate import evaluate

import torch.optim as optim


# Kaggle -> NIH
k_train = load_kaggle(DATASETS["kaggle"]["train"])
nih_test = load_nih(DATASETS["nih"]["test"], DATASETS["nih"]["csv"])

model = get_model()
criterion = get_loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model, k_train, optimizer, criterion)

print("Kaggle → NIH")

evaluate(model, nih_test)


# NIH -> COVID
nih_train = load_nih(DATASETS["nih"]["train"], DATASETS["nih"]["csv"])
covid_test = load_covid(DATASETS["covid"]["test"])

model = get_model()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model, nih_train, optimizer, criterion)

print("NIH → COVID")

evaluate(model, covid_test)


# COVID -> Kaggle
covid_train = load_covid(DATASETS["covid"]["train"])
k_test = load_kaggle(DATASETS["kaggle"]["test"])

model = get_model()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model, covid_train, optimizer, criterion)

print("COVID → Kaggle")

evaluate(model, k_test)