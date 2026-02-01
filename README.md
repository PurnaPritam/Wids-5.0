# Wids-5.0 - Image Deblurring 

This repository serves as a comprehensive log of my three-week deep learning coursework. The curriculum progressed from fundamental neural network architectures to advanced computer vision applications, specifically image super-resolution.

Below is a detailed summary of the methodologies implemented and the performance metrics achieved for each project.

## Week 1: Foundations with Neural Networks (NN)
**Objective:** Implement a dense Feed-Forward Network to classify handwritten digits from the MNIST dataset.

I developed a Multi-Layer Perceptron (MLP) architecture designed to process flattened 784-dimensional vectors (28x28 images). The topology consisted of three fully connected layers (784→256→128→10) using ReLU activation functions, incorporated with a Dropout rate of 0.3 to mitigate overfitting during training.

**Outcomes:**
- The model demonstrated solid convergence over 5 epochs with consistent loss reduction, achieving a **Test Accuracy of 97.32%** on the 10,000-sample MNIST test set.
- In testing, I evaluated the model on one sample from each digit class (0-9), where the model correctly identified **10 out of 10** instances with confidence scores averaging above 95%.
- *Analysis:* While the accuracy was satisfactory for a baseline, flattening the image data discarded critical spatial hierarchies, suggesting that a convolutional approach would yield better generalization.

## Week 2: Convolutional Neural Networks (CNN)
**Objective:** Leverage spatial feature extraction to improve classification accuracy.

Transitioning to a Convolutional Neural Network allowed the model to process the image data in its native 2D format. I designed an architecture featuring three convolutional blocks (32→64→128 filters)—each comprising Conv2d, BatchNorm2d, ReLU, and MaxPooling layers—to systematically extract local features like edges and curves before passing them to three dense layers (256→128→10).

**Outcomes:**
- **Performance Gain:** The introduction of convolutional layers resulted in a significant performance boost, raising the **Test Accuracy to 99.20%** on the standard MNIST test set.
- **Custom Image Testing:** To validate real-world performance, I tested the trained model on 10 handwritten digits (0-9) that I personally created and photographed. The model successfully classified **all 10 out of 10** of these images, with confidence scores ranging from 92.21% to 100.00%.
- *Analysis:* The CNN proved far superior in handling slight spatial variances and handwriting styles compared to the standard MLP, demonstrating a 1.88 percentage point improvement in test accuracy and perfect performance on custom samples.

## Week 3: Image Restoration with SRCNN
**Objective:** Implement the Super-Resolution CNN (SRCNN) architecture for image deblurring and upscaling.

This week focused on the task of mapping low-resolution (blurred) images to high-resolution targets using the CIFAR-10 dataset. I implemented the canonical SRCNN architecture with a 9-1-5 kernel configuration (Conv2d: 3→64→32→3 channels) to learn the non-linear mapping required to sharpen images and restore high-frequency details. The model was trained for 100 epochs on 5,000 CIFAR-10 images with synthetically introduced blur (4x downsampling followed by bicubic interpolation).

**Outcomes:**
- **Metric:** I used Peak Signal-to-Noise Ratio (PSNR) to measure reconstruction quality in decibels (dB).
- **Results:** Tested on 5 held-out CIFAR-10 test images, the trained model achieved an **average improvement of 2.65 dB** over the baseline bicubic interpolation (blurred images: ~24.5 dB, deblurred images: ~27.15 dB).
- *Analysis:* Visual inspection confirmed that the network successfully learned to reduce blur artifacts, producing output images with noticeably sharper edges and clearer textures than the inputs. The consistent 2-3 dB improvement across all test samples validated the model's generalization capability.

## Week 4: Image Deblurring with DeblurGAN
**Objective:** Implement a Generative Adversarial Network (GAN) architecture for motion blur removal and image restoration.

This week focused on adversarial training for image-to-image translation, specifically deblurring motion-blurred images using the STL-10 dataset (96x96 high-quality images). I implemented a DeblurGAN architecture consisting of a U-Net Generator with skip connections (Conv2d: 3→64→128→256→512 channels in the encoder, mirrored in the decoder) and a Multi-Scale PatchGAN Discriminator operating at two resolutions. The model was trained for 100 epochs on 5,000 STL-10 images with synthetically introduced motion blur (variable kernel sizes 7-15 with random angles 0-180°). The loss function combined L1 reconstruction loss (λ=100), VGG perceptual loss (λ=0.01), and adversarial loss (λ=1.0).

**Outcomes:**
- **Metric:** I used Peak Signal-to-Noise Ratio (PSNR) to measure reconstruction quality in decibels (dB).
- **Results:** Tested on 10 held-out STL-10 test images, the trained model achieved an **average PSNR improvement of ~3-4 dB** over the baseline motion-blurred images, with the Generator and Discriminator losses stabilizing after approximately 50 epochs.
- *Analysis:* Visual inspection confirmed that the network successfully learned to remove motion blur artifacts, producing output images with noticeably sharper edges and restored high-frequency details compared to the blurred inputs. The U-Net skip connections proved essential for preserving fine structural details, while the multi-scale discriminator ensured globally coherent deblurring across different image regions.
