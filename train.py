import argparse
import mlflow
import json
import torch
import joblib
import torchmetric
from cnn_model import RHSCNN, train_loader, validation_loader

def get_flags_passed_in_from_terminal():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r')
    args = parser.parse_args()
    return args

def get_train_features_labels():
    for train_features, train_labels in train_loader:
        return train_features, train_labels

def get_val_features_labels():
    for val_features, val_labels in validation_loader:
        return val_features, val_labels

train_features, train_labels = get_train_features_labels()
val_features, val_labels = get_val_features_labels()
# feature_names = [train_features, val_features]

args = get_flags_passed_in_from_terminal()
print(args)

# SET EXPERIMENT
with mlflow.start_run():

    # LOG ARTIFACT
    
    # FIT MODEL
    model = RHSCNN()
    train_output = model(train_features)

    # LOG MODEL
    joblib.dump(model, 'saved_model.joblib')

    # LOG METRIC
    metric = torchmetric.Accuracy()
    score = metric(train_output, train_labels)
    mlflow.log_metric('Accuracy', score)