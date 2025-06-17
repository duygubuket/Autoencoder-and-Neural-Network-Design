import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Encoder(nn.Module):
    """Encoder network that compresses 28x28 images to latent space"""
    def __init__(self, input_dim=784, latent_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        # Flatten the input if it's not already flattened
        x = x.view(x.size(0), -1)
        return self.encoder(x)

class Decoder(nn.Module):
    """Decoder network that reconstructs images from latent space"""
    def __init__(self, latent_dim=64, output_dim=784):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Sigmoid to ensure output is in [0, 1]
        )
    
    def forward(self, x):
        x = self.decoder(x)
        # Reshape to image format (28x28)
        return x.view(x.size(0), 1, 28, 28)

class Autoencoder(nn.Module):
    """Complete Autoencoder combining Encoder and Decoder"""
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
    
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        # Decode
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Generate from latent code"""
        return self.decoder(z)

def load_fashion_mnist():
    """Load and prepare Fashion-MNIST dataset"""
    # Transform to normalize data to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Note: Fashion-MNIST is already normalized to [0, 1] with ToTensor()
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def train_autoencoder(model, train_loader, test_loader, epochs=25, lr=0.001):
    """Train the autoencoder"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                reconstructed = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def visualize_reconstructions(model, test_loader, num_samples=10):
    """Visualize original vs reconstructed images"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # Move to CPU and convert to numpy
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
    
    for i in range(num_samples):
        # Original images
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return images, reconstructed

def generate_new_samples(model, num_samples=10, latent_dim=64):
    """Generate new samples by sampling from latent space"""
    model.eval()
    
    with torch.no_grad():
        # Sample random points from latent space (Gaussian noise)
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate images using decoder
        generated = model.decode(z)
        generated = generated.cpu()
    
    # Visualize generated samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(generated[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Generated {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return generated

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_reconstruction_loss(model, test_loader):
    """Calculate and report reconstruction loss on test set"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    print(f"\nTest Set Reconstruction Loss (MSE): {avg_loss:.6f}")
    return avg_loss

def main():
    """Main execution function"""
    print("Fashion-MNIST Autoencoder Implementation")
    print("="*50)
    
    # 1. Dataset Preparation
    print("1. Loading Fashion-MNIST dataset...")
    train_loader, test_loader = load_fashion_mnist()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Model Creation
    print("\n2. Creating Autoencoder model...")
    latent_dim = 64  # Size of latent space
    model = Autoencoder(latent_dim=latent_dim).to(device)
    
    # Print model architecture
    print(f"Latent dimension: {latent_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Training
    print("\n3. Training the Autoencoder...")
    train_losses, val_losses = train_autoencoder(model, train_loader, test_loader, epochs=25)
    
    # 4. Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(train_losses, val_losses)
    
    # 5. Evaluation
    print("\n5. Evaluating the Autoencoder...")
    test_loss = evaluate_reconstruction_loss(model, test_loader)
    
    # 6. Visualize reconstructions
    print("\n6. Visualizing reconstructions...")
    original, reconstructed = visualize_reconstructions(model, test_loader, num_samples=10)
    
    # 7. Generate new samples
    print("\n7. Generating new samples...")
    generated_samples = generate_new_samples(model, num_samples=10, latent_dim=latent_dim)
    
    # 8. Fashion-MNIST class names for reference
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print(f"\nFashion-MNIST classes: {class_names}")
    print("\nTraining completed successfully!")
    
    # Save the trained model
    torch.save(model.state_dict(), 'fashion_mnist_autoencoder.pth')
    print("Model saved as 'fashion_mnist_autoencoder.pth'")

if __name__ == "__main__":
    main()