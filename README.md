# Autoencoder-and-Neural-Network-Design

## 📚 Project Overview
This repository contains two deep learning models trained on the **Fashion-MNIST** dataset:

- 🧬 An **Autoencoder** for unsupervised image reconstruction and generation.
- 🧠 A **Feedforward Neural Network Classifier** for supervised multi-class classification.

Both models are implemented **from scratch** using PyTorch and support complete training, evaluation, and visualization pipelines.

---

## 📂 File Structure

```
fashion-mnist-hw2/
├── main.py                     # Entry point script
├── utils.py                    # Autoencoder model, loaders, visualizations
├── classifier_utils.py         # Classifier model, training & metrics
├── README.md                   # This file
└── images/                     # Plots and sample outputs (optional)

## 🛠️ Autoencoder Details

### 🔧 Architecture

- **Encoder:**  
  784 → 512 → 256 → 128 → **64 (Latent Space)**
- **Decoder:**  
  64 → 128 → 256 → 512 → 784

### 🧪 Hyperparameters

| Parameter         | Value         |
|------------------|---------------|
| Latent Dimension | 64            |
| Optimizer        | Adam          |
| Learning Rate    | 0.001         |
| Batch Size       | 128           |
| Epochs           | 25            |
| Loss Function    | MSE           |

### 📊 Performance

- **Final MSE on Test Set:** 0.009931
- **Reconstruction Accuracy:** ~99.01%
- **Visual Quality:** High fidelity with clear structural preservation

### 🔍 Generation Results

- Random latent vectors were sampled from N(0,1)
- Generated images captured garment structure, though blurry and less diverse
- Suggested improvements: **Variational Autoencoder**, **regularization**, and **FID evaluation**

---

## 🧮 Classifier Details

### 🔧 Architecture

- Input: 784 → 512 → 256 → 128 → **Softmax (10 classes)**

### 🧪 Hyperparameters

| Parameter          | Value                    |
|-------------------|--------------------------|
| Loss Function      | CrossEntropyLoss         |
| Optimizer          | Adam                     |
| Learning Rate      | 0.001                    |
| Epochs             | 50                       |
| Dropout Rate       | 0.3                      |
| Weight Decay       | 1e-4                     |
| Scheduler          | StepLR (step=20, γ=0.5)  |

### ⚙️ Regularization

- **Batch Normalization** after each layer
- **Dropout (30%)** to prevent overfitting
- **L2 Regularization**

---

## 📈 Performance

| Metric               | Value     |
|----------------------|-----------|
| Final Test Accuracy  | 90.07%    |
| Training Accuracy    | 96.32%    |
| Best Validation Acc. | 90.94%    |
| F1-Score (Macro Avg) | **0.90**  |

### 🔍 Confusion Matrix Observations

- **Well-Classified:** Trouser, Bag, Sandal, Ankle Boot
- **Most Confused:** T-shirt ↔ Shirt, Pullover ↔ Coat
- **Causes of Error:** shape ambiguity, texture similarity, unusual orientations

---

## 📷 Visualizations

- **Train/Val Loss Curves**
- **Reconstructed vs Original Samples**
- **Generated Images**
- **Confusion Matrix**
- **Misclassified Examples**

> All visuals are included in the `images/` folder and `report.pdf`.

---

## 📌 Key Insights

- The autoencoder achieved strong reconstructions but weaker generative performance.
- The classifier was effective overall, with ~90% accuracy and challenges in visually similar classes.
- Regularization (dropout, weight decay) helped limit overfitting, but further tuning (e.g., deeper architectures, data augmentation) may enhance results.

---

## 🧑‍💻 Author

**Duygu Buket Bıyık**  
📧 [duygubuketbiyik@gmail.com](mailto:duygubuketbiyik@gmail.com)  
🌐 [GitHub](https://github.com/duygubuket) | [LinkedIn](https://www.linkedin.com/in/duygubuketbiyik)

---

## 📂 License

This project is part of a university homework assignment and intended for academic use only.
