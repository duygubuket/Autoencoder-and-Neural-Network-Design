# classifier_utils.py
# Training and evaluation utilities for Feedforward Neural Network classifier

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Accuracy calculation
def compute_accuracy(y_true, y_pred):
    correct = torch.sum(y_true == y_pred).item()
    total = len(y_true)
    return 100.0 * correct / total

# Confusion matrix calculation
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm.numpy()

# Precision, Recall, F1

def compute_classification_metrics(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    return precision, recall, f1

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        train_acc = 100. * correct / total
        history['train_losses'].append(train_loss / len(train_loader))
        history['train_accuracies'].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                _, pred = output.max(1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)

        val_acc = 100. * val_correct / val_total
        history['val_losses'].append(val_loss / len(val_loader))
        history['val_accuracies'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    model.load_state_dict(best_model)
    return history

# Evaluation

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    acc = compute_accuracy(torch.tensor(all_labels), torch.tensor(all_preds))
    return acc, np.array(all_preds), np.array(all_labels)

# Reporting

def print_classification_report_from_scratch(precision, recall, f1, class_names):
    print("\nClassification Report")
    print("="*60)
    print(f"{'Class':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
    print("-"*60)
    for i, name in enumerate(class_names):
        print(f"{name:<15}{precision[i]:<10.2f}{recall[i]:<10.2f}{f1[i]:<10.2f}")
    print("="*60)
    print(f"Macro Avg    {np.mean(precision):.2f}     {np.mean(recall):.2f}     {np.mean(f1):.2f}")
