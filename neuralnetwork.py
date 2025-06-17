import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fashion-MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Dataset preparation
def load_fashion_mnist():
    """Load and preprocess Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST normalization
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Feedforward Neural Network Architecture (FROM SCRATCH)
class FeedforwardNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout_rate=0.3):
        super(FeedforwardNN, self).__init__()
        
        # Create layers list
        layers = []
        prev_size = input_size
        
        # Hidden layers with batch normalization and dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights FROM SCRATCH
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot uniform initialization
                fan_in = module.in_features
                fan_out = module.out_features
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                module.weight.data.uniform_(-bound, bound)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)  # Reshape from (batch_size, 1, 28, 28) to (batch_size, 784)
        return self.network(x)

# FROM SCRATCH: Confusion Matrix calculation
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix from scratch"""
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        confusion_matrix[true_label][pred_label] += 1
    
    return confusion_matrix.numpy()

# FROM SCRATCH: Classification metrics calculation
def compute_classification_metrics(confusion_matrix):
    """Compute precision, recall, and F1-score from confusion matrix from scratch"""
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True Positives
        tp = confusion_matrix[i, i]
        
        # False Positives (sum of column i, excluding diagonal)
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False Negatives (sum of row i, excluding diagonal)
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Precision = TP / (TP + FP)
        if (tp + fp) > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0
        
        # Recall = TP / (TP + FN)
        if (tp + fn) > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0.0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision[i] + recall[i]) > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_score[i] = 0.0
    
    return precision, recall, f1_score

# FROM SCRATCH: Accuracy calculation
def compute_accuracy(y_true, y_pred):
    """Compute accuracy from scratch"""
    correct = torch.sum(y_true == y_pred).item()
    total = len(y_true)
    return 100.0 * correct / total

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the feedforward neural network"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Track metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_train_preds.extend(predicted.cpu())
            all_train_targets.extend(targets.cpu())
        
        # Calculate training accuracy FROM SCRATCH
        train_acc = compute_accuracy(torch.tensor(all_train_targets), torch.tensor(all_train_preds))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu())
                all_val_targets.extend(targets.cpu())
        
        # Calculate validation accuracy FROM SCRATCH
        val_acc = compute_accuracy(torch.tensor(all_val_targets), torch.tensor(all_val_preds))
        
        # Calculate average losses
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'training_time': training_time
    }

# Evaluation function FROM SCRATCH
def evaluate_model(model, test_loader):
    """Evaluate the model on test dataset"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu())
            all_targets.extend(targets.cpu())
    
    # Convert to tensors
    all_predictions = torch.tensor(all_predictions)
    all_targets = torch.tensor(all_targets)
    
    # Calculate accuracy FROM SCRATCH
    accuracy = compute_accuracy(all_targets, all_predictions)
    
    return accuracy, all_predictions.numpy(), all_targets.numpy()

# Visualization functions
def plot_training_curves(history):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_accuracies'], label='Training Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_from_scratch(confusion_matrix, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap manually
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, str(confusion_matrix[i, j]),
                    horizontalalignment="center", 
                    verticalalignment="center",
                    color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names, rotation=0)
    plt.tight_layout()
    plt.show()

def print_classification_report_from_scratch(precision, recall, f1_score, class_names):
    """Print classification report"""
    print("\nDetailed Classification Report:")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<10.2f} {recall[i]:<10.2f} {f1_score[i]:<10.2f}")
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)
    
    print("-" * 70)
    print(f"{'Macro Avg':<15} {macro_precision:<10.2f} {macro_recall:<10.2f} {macro_f1:<10.2f}")
    print("-" * 70)

def show_misclassified_examples(model, test_loader, num_examples=8):
    """Show examples of misclassified images"""
    model.eval()
    misclassified_images = []
    misclassified_preds = []
    misclassified_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            # Find misclassified samples
            mask = predicted != targets
            if mask.sum() > 0:
                misclassified_images.extend(data[mask].cpu())
                misclassified_preds.extend(predicted[mask].cpu())
                misclassified_targets.extend(targets[mask].cpu())
                
                if len(misclassified_images) >= num_examples:
                    break
    
    # Plot misclassified examples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_examples, len(misclassified_images))):
        img = misclassified_images[i].squeeze()
        pred = misclassified_preds[i]
        target = misclassified_targets[i]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {class_names[target]}\nPred: {class_names[pred]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    """Main function to run the complete pipeline"""
    print("Fashion-MNIST Feedforward Neural Network Classification")
    print("=" * 80)
    
    # Load data
    print("Loading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader = load_fashion_mnist()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating feedforward neural network")
    model = FeedforwardNN(
        input_size=784,
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        dropout_rate=0.3
    ).to(device)
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n" + "="*80)
    history = train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test dataset...")
    test_accuracy, predictions, targets = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Compute confusion matrix FROM SCRATCH
    confusion_matrix = compute_confusion_matrix(targets, predictions, num_classes=10)
    
    # Compute classification metrics FROM SCRATCH
    precision, recall, f1_score = compute_classification_metrics(confusion_matrix)
    
    # Print classification report FROM SCRATCH
    print_classification_report_from_scratch(precision, recall, f1_score, class_names)
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_training_curves(history)
    plot_confusion_matrix_from_scratch(confusion_matrix, class_names)
    show_misclassified_examples(model, test_loader)
    
    # Print summary report
    print("\n" + "="*80)
    print("FINAL REPORT SUMMARY")
    print("="*80)
    print(f"Architecture: 3-layer feedforward neural network")
    print(f"Hidden layers: [512, 256, 128] neurons")
    print(f"Activation: ReLU")
    print(f"Regularization: Batch Normalization + Dropout (0.3)")
    print(f"Optimizer: Adam with weight decay (1e-4)")
    print(f"Learning rate: 0.001 with StepLR scheduler")
    print(f"Training time: {history['training_time']:.2f} seconds")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Total parameters: {total_params:,}")
    
    return model, history, test_accuracy

if __name__ == "__main__":
    model, history, test_accuracy = main()