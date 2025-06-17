# main.py

from utils import *

if __name__ == "__main__":
    print("Running Autoencoder pipeline...\n" + "="*60)

    # Load dataset for autoencoder (un-normalized)
    train_loader, val_loader, test_loader = load_fashion_mnist(normalize=False)

    # Initialize model
    latent_dim = 64
    autoencoder = Autoencoder(latent_dim).to(device)
    print(f"Autoencoder total parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")

    # Train autoencoder
    train_losses, val_losses = train_autoencoder(autoencoder, train_loader, test_loader, epochs=25, lr=0.001)
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)

    # Evaluate test loss
    evaluate_reconstruction_loss(autoencoder, test_loader)

    # Show reconstructions
    show_reconstructions(autoencoder, test_loader)

    # Generate samples
    generate_samples(autoencoder, latent_dim=latent_dim)

    # Save model
    torch.save(autoencoder.state_dict(), "autoencoder_fashionmnist.pth")
    print("Autoencoder model saved as 'autoencoder_fashionmnist.pth'")

    print("\n" + "="*60)
    print("Running Classifier pipeline...\n" + "="*60)

    # Load dataset for classifier (normalized)
    train_loader, val_loader, test_loader = load_fashion_mnist(normalize=True)

    # Initialize classifier
    classifier = FeedforwardNN().to(device)

    # Train classifier
    results = train_model(classifier, train_loader, val_loader, num_epochs=50)
    
    # Evaluate classifier
    test_acc, y_pred, y_true = evaluate_model(classifier, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Confusion matrix & metrics
    cm = compute_confusion_matrix(y_true, y_pred)
    prec, rec, f1 = compute_classification_metrics(cm)
    print_classification_report_from_scratch(prec, rec, f1, class_names)

    # Visualizations
    plot_training_curves(results)
    plot_confusion_matrix_from_scratch(cm, class_names)
    show_misclassified_examples(classifier, test_loader)

    # Save classifier
    torch.save(classifier.state_dict(), "classifier_fashionmnist.pth")
    print("Classifier model saved as 'classifier_fashionmnist.pth'")
