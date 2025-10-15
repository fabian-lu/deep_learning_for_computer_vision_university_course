# Deep Learning for Computer Vision - University Course

This repository contains homework assignments from a university course on Deep Learning for Computer Vision, completed in Winter Term 2024/25.

## Course Information

- **Lecture by:** Constantin Pape
- **Institution:** University Course
- **Term:** Winter Term 2024/25

## Course Content

The assignments cover fundamental and advanced deep learning concepts for computer vision, implemented in Python using PyTorch. All exercises were completed on Kaggle notebooks with GPU acceleration.

### Homework 1: CIFAR-10, MLP, Overfitting and Regularization
**Topics:**
- Working with CIFAR-10 dataset
- Multi-Layer Perceptron (MLP) implementation
- Model fitting and training procedures
- Observing and understanding overfitting
- Early stopping implementation
- Hyperparameter exploration and tuning
- Architecture variations to improve performance

**Key Concepts:** Image classification, regularization techniques, model evaluation

### Homework 2: CNNs, ResNets and LR-Scheduling
**Topics:**
- Convolutional Neural Networks (CNNs)
- ResNet architecture implementation
- Learning rate scheduling strategies
- Advanced training techniques
- Comparative analysis of different architectures

**Key Concepts:** Convolution operations, residual connections, optimization strategies

### Homework 3: COVID-19 Classification using Transfer Learning
**Topics:**
- Medical image classification from Chest X-Ray images
- Multi-class classification (COVID-19, Pneumonia, Healthy)
- Training ResNets from scratch
- Transfer learning with ImageNet pre-trained models
- Transfer learning with RadImageNet (specialized radiology dataset)
- Comparing different transfer learning approaches
- Fine-tuning strategies

**Dataset:** Subset of COVID-19 Chest X-Ray Kaggle dataset

**Key Concepts:** Transfer learning, domain adaptation, medical imaging, model comparison

### Homework 4: Segmentation & Denoising with U-Net
**Topics:**
- U-Net architecture for image segmentation
- Image denoising techniques
- Encoder-decoder architectures
- Semantic segmentation tasks
- Noise reduction in images

**Key Concepts:** Segmentation, U-Net, skip connections, pixel-wise prediction

### Homework 5: Variational Autoencoders (VAE)
**Topics:**
- Variational Autoencoder architecture
- Generative modeling
- Latent space representation
- Encoder-decoder structure for VAEs
- Reconstruction and generation

**Key Concepts:** Generative models, latent variables, probabilistic modeling, unsupervised learning

## Technologies Used

- **Python 3**
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision utilities and models
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **Jupyter Notebooks** - Interactive development
- **Kaggle** - GPU-accelerated notebook environment

## Repository Structure

```
Deep_Learning_university_course/
├── homework_1.ipynb    # CIFAR-10, MLP, Overfitting
├── homework_2.ipynb    # CNNs, ResNets, LR-Scheduling
├── homework_3.ipynb    # Transfer Learning for Medical Imaging
├── homework_4.ipynb    # U-Net Segmentation & Denoising
├── homework_5.ipynb    # Variational Autoencoders
└── README.md
```

## Key Learning Objectives

- Understanding fundamental deep learning architectures (MLPs, CNNs, ResNets, U-Net, VAEs)
- Implementing and training neural networks with PyTorch
- Recognizing and addressing overfitting through regularization and early stopping
- Applying transfer learning to new domains and datasets
- Working with medical imaging data
- Implementing segmentation and generative models
- Hyperparameter tuning and optimization strategies
- Evaluating model performance and making architectural improvements

## Course Highlights

This course provided hands-on experience with:
- Building models from scratch and understanding their inner workings
- Practical applications to real-world datasets (CIFAR-10, medical X-rays)
- State-of-the-art architectures (ResNet, U-Net, VAE)
- Transfer learning with both general (ImageNet) and domain-specific (RadImageNet) pre-trained models
- Complete ML pipeline from data loading to model evaluation
