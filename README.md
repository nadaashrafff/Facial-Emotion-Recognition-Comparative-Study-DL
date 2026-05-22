# Comparative Study of Deep Learning Architectures for Facial Emotion Recognition Using FERPlus

## Project Overview

This project presents a comparative study of deep learning architectures for **Facial Emotion Recognition (FER)** using the **FERPlus dataset**.

Facial Emotion Recognition is a computer vision task that aims to classify human emotions from facial expressions. This project compares multiple CNN-based and transfer learning models under a unified experimental setup to evaluate their accuracy, generalization ability, stability, and computational efficiency.

The study evaluates both models trained from scratch and pretrained transfer learning models to understand the trade-offs between model complexity, accuracy, and deployment efficiency.

---

## Research Title

**A Comparative Study of Deep Learning Architectures for Facial Emotion Recognition Using the FERPlus Dataset**

---

## Authors

- Nada Ashraf
- Aly Zaki
- Omar Bayoumi
- Ahmed Waleed

**Faculty:** Computing and Information Systems  
**Track:** Artificial Intelligence  
**University:** Egypt University of Informatics  
**Supervisor:** Dr. Mona Soliman  

---

## Problem Statement

Facial emotion recognition remains challenging because facial expressions can be subtle, ambiguous, and visually similar across different emotion classes.

Common challenges include:

- Subtle facial muscle movements
- Class imbalance
- Noisy or ambiguous labels
- Similar emotion pairs such as Fear and Surprise
- Similar emotion pairs such as Neutral and Sad
- Variations in lighting, pose, and facial occlusion

This project investigates how different deep learning architectures perform on FERPlus and which models provide the best balance between accuracy, stability, and efficiency.

---

## Dataset

The project uses the **FERPlus dataset**, an improved version of the FER2013 dataset.

FERPlus improves label reliability by using crowdsourced annotations, where each image is labeled by multiple human annotators.

### Emotion Classes

The dataset includes facial expressions such as:

- Angry
- Contempt
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Some experiments use all 8 classes, while others use a reduced 7-class setup by removing the highly imbalanced Contempt class.

---

## Project Objective

The main objective of this project is to compare multiple deep learning models for facial emotion recognition using a consistent dataset and evaluation protocol.

The project aims to:

- Train and evaluate CNN models from scratch.
- Evaluate transfer learning models pretrained on ImageNet.
- Compare model accuracy, stability, and efficiency.
- Analyze overfitting and generalization behavior.
- Study class-wise performance using confusion matrices.
- Identify common misclassification patterns.
- Provide practical insights for selecting FER architectures.

---

## Models Evaluated

A total of eight deep learning architectures were evaluated.

## 1. Models Trained From Scratch

### GAP-Regularized Stronger CNN

A stronger CNN architecture using Global Average Pooling to reduce parameters and improve generalization.

**Key Features:**

- 224 × 224 × 3 input
- Four convolutional blocks
- Batch Normalization
- ReLU activation
- MaxPooling
- Global Average Pooling
- Dropout and L2 regularization

**Test Accuracy:** 71.09%

---

### Baseline CNN

A lightweight CNN trained from scratch using native FERPlus grayscale images.

**Key Features:**

- 48 × 48 × 1 grayscale input
- Three convolutional stages
- Batch Normalization
- Dropout
- Fully connected classification head

**Test Accuracy:** 76.00%

---

### Custom Inception-Based CNN

A custom CNN architecture that uses Inception-style modules to capture multi-scale facial features.

**Key Features:**

- 48 × 48 × 1 grayscale input
- Parallel 1×1, 3×3, and 5×5 convolution branches
- Global Average Pooling
- Batch Normalization
- Dropout
- Label smoothing

**Test Accuracy:** 72.21%

---

### Custom Inception–ResNet CNN

A custom model combining Inception-style multi-scale feature extraction with residual connections.

**Key Features:**

- 48 × 48 × 1 grayscale input
- 7-class setup
- Duplicate and leakage removal
- Inception–ResNet blocks
- SpatialDropout
- Global Average Pooling
- Cosine annealing learning rate schedule

**Test Accuracy:** 76.29%

This was the strongest overall model in terms of accuracy and stability among the evaluated models.

---

## 2. Transfer Learning Models

### ResNet50 Feature Extraction Model

A transfer learning model using ResNet50 pretrained on ImageNet as a frozen feature extractor.

**Key Features:**

- Grayscale to RGB conversion
- 224 × 224 × 3 input
- Frozen ResNet50 backbone
- Global Average Pooling
- Dense classification head

**Test Accuracy:** 69.41%

---

### EfficientNetB3 Feature Extraction Model

A transfer learning model using EfficientNetB3 pretrained on ImageNet.

**Key Features:**

- 224 × 224 × 3 input
- Frozen EfficientNetB3 backbone
- Global Average Pooling
- Batch Normalization
- Dropout
- Efficient parameter scaling

**Test Accuracy:** Approximately 70%

Fine-tuning was tested, but it caused overfitting, so the frozen-backbone phase was considered more reliable.

---

### DenseNet121 Transfer Learning Model

A transfer learning model using DenseNet121 pretrained on ImageNet with a two-phase training strategy.

**Key Features:**

- 96 × 96 input
- 7-class setup
- Duplicate and leakage removal using perceptual hashing
- Frozen backbone in Phase 1
- Top 15% of layers unfrozen in Phase 2
- Dense classification head

**Test Accuracy:** 67.83%

---

### MobileNetV2 Feature Extraction Model

A lightweight transfer learning model using MobileNetV2 pretrained on ImageNet.

**Key Features:**

- 224 × 224 × 3 input
- Frozen MobileNetV2 backbone
- Depthwise separable convolutions
- Inverted residual blocks
- Low parameter count
- High efficiency

**Test Accuracy:** Approximately 68%

MobileNetV2 offers strong efficiency and is suitable for resource-constrained deployment.

---

## Performance Summary

### Models Trained From Scratch

| Model | Input | Classes | Test Accuracy | Stability |
|---|---:|---:|---:|---|
| Baseline CNN | 48 × 48 | 8 | 76.00% | Low |
| GAP-SCNN | 224 × 224 | 8 | 71.09% | High |
| Inception-CNN | 48 × 48 | 8 | 72.21% | Medium |
| Inception–ResNet CNN | 48 × 48 | 7 | 76.29% | High |

---

### Transfer Learning Models

| Model | Input | Classes | Test Accuracy | Efficiency |
|---|---:|---:|---:|---|
| ResNet50 | 224 × 224 | 8 | 69.41% | Medium |
| EfficientNetB3 | 224 × 224 | 8 | ~70% | High |
| DenseNet121 | 96 × 96 | 7 | 67.83% | Medium |
| MobileNetV2 | 224 × 224 | 8 | ~68% | Very High |

---

## Key Findings

The study found that:

- Carefully regularized models trained from scratch achieved the highest test accuracy.
- The Inception–ResNet CNN achieved the best overall balance between accuracy and generalization.
- Transfer learning models showed strong training stability but had a lower accuracy ceiling than the best from-scratch models.
- Lightweight transfer learning models such as MobileNetV2 offer useful efficiency advantages.
- Fine-tuning pretrained models too aggressively can cause overfitting.
- Common confusion patterns appeared across models, especially:
  - Fear vs Surprise
  - Neutral vs Sad
  - Angry vs Disgust
- Removing highly imbalanced classes such as Contempt can improve training stability.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix analysis
- Training and validation accuracy curves
- Training and validation loss curves
- Generalization gap analysis
- Grad-CAM interpretability for selected models

---

## Model Interpretability

Grad-CAM visualizations were used for selected architectures to understand where models focus when making predictions.

Correct predictions generally focused on meaningful facial regions such as:

- Eyes
- Eyebrows
- Mouth
- Facial expression regions

Misclassified samples often showed overlapping attention patterns between visually similar emotions.

---

## Main Conclusion

No single model dominated across all evaluation dimensions.

The **Inception–ResNet CNN** achieved the best overall balance between accuracy and stability, while transfer learning models provided better training stability and lower sensitivity to hyperparameter tuning.

For deployment-oriented scenarios, lightweight models such as **MobileNetV2** provide an attractive balance between efficiency and robustness.





---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Grad-CAM
- CNN architectures
- Transfer learning
- FERPlus dataset

---

## Possible Future Work

Future work may include:

- Testing additional architectures such as Vision Transformers.
- Applying ensemble learning between the best models.
- Improving minority-class performance using advanced class balancing.
- Testing on real-world facial expression datasets.
- Applying stronger data cleaning and label refinement.
- Deploying lightweight models on mobile or edge devices.
- Using attention mechanisms for better emotion localization.

---

## Academic Context

This project was conducted as part of the **Deep Learning Course** at the Faculty of Computing and Information Systems, Egypt University of Informatics.

---

## Acknowledgements

The authors would like to express their sincere gratitude to **Dr. Mona Soliman** for her guidance, support, and valuable feedback throughout the development of this project.
