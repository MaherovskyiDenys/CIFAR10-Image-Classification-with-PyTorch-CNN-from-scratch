# CIFAR-10 Image Classification with PyTorch (CNN from scratch)

A simple CNN built from scratch for educational purposes

## Description

This project focuses on building a solid fundamental understanding of:
- Data augmentation
- Dropout
- Batch normalization

The model achieves ~89% test accuracy on the CIFAR-10 test dataset (varies slightly with random seed and augmentation).

Note: This project intentionally avoids advanced techniques such as learning rate schedulers or global average pooling
to find out the limits of the CNN architecture using only its core components

## Structure
    
[conv → bn → relu] →  
[conv → bn → relu → max_pool] →  
[conv → bn → relu → max_pool] →  
[conv → bn → relu → max_pool → flatten] →  
FC

![diagram.png](assets%2Fdiagram.png)

## Getting Started

### Dependencies

* torchvision>=0.24.1
* matplotlib>=3.10
* torch>=2.9

### Executing program

Clone the repository and navigate to the project root directory

### Training the model

If the model has not been trained yet or pre-trained weights are not available, run:

```shell
python -m network.train 
```
This will:
- Train the CNN on the CIFAR-10 dataset
- Save the trained model weights
- Display loss and accuracy curves after training completes

### Evaluating a custom image

To run inference on a custom image:

Acceptable formats: JPEG, PNG or GIF 

```shell
python -m network.evaluate path/to/image.jpg  --show
```
The image will be automatically resized to 32×32 and normalized using CIFAR-10 statistics before evaluation
Use (optional) `--show` flag to display your already resized image

Several example images are available in the test/img directory

### Evaluating model performance

The test folder contains evaluation scripts that measures model performance on the CIFAR-10 test set

For example, `test/test_model` evaluates the saved model and reports test loss and accuracy

## Results (CIFAR-10)

- Test Accuracy: ~89%
- Training Accuracy: ~94%
- No learning-rate scheduling
- No weight decay
- No global average pooling

This project intentionally focuses on core CNN architecture,
data augmentation, and clean training/evaluation pipelines.

## Things I Learned - (Personal)

### Data & augmentation
- Data augmentation is an important tool in computer vision, however more is not always better
- CIFAR-10 is a small dataset with (32×32) small images, so augmentation has to be gentle and targeted
- Rotations `v2.RandomRotation(10)` made performance worse
- Proper normalization matters. CIFAR-10 mean and standard deviation were computed from the training set, see acknowledgments section
- More augmentation ≠ better generalization, most likely will show worse performance

### Architecture decisions
- Removing `nn.MaxPool2d(2)` in the first convolutional block was necessary because aggressive down-scaling harms small images (32x32)
- Increasing depth to 4 convolutional blocks improved feature extraction
- Dropout was applied **only** to fully connected layers, using it in convolutional layers hurt performance
- `nn.BatchNorm2d` **significantly** stabilized training and improved convergence
- BatchNorm allowed reducing training epochs from 80 → 50, and decrease dropout probability to 0.1 while improving accuracy by ~1.4%
- BatchNorm is most effective when used before activation and pooling layer
- `nn.LazyLinear` handy module if trying to avoid hard-coded layers

### Training dynamics
- More epochs ≠ better model, accuracy was often higher before the final epoch
- The best model is not necessarily the last epoch
- Logging both train and test loss/accuracy is **essential** to diagnose overfitting
- Learning curves provided clearer understanding than raw accuracy numbers alone
![loss_vs_accuracy.png](assets%2Floss_vs_accuracy.png)

### PyTorch engineering practices
- Separating model architecture from optimizer, learning rate, and training logic improved clarity and flexibility, overall very good practice
- Loss functions or optimizer should not be hard-coded into the model
- Using `nn.Sequential` helps produce idiomatic and readable PyTorch code
- Correct usage of `model.train()` and `model.eval()` is critical for reliable training and evaluation
- Debugging CUDA device mismatches reinforced careful tensor and model placement

## Training setup
* Dataset: CIFAR-10
* Optimizer: Adam (lr=1e-03)
* Epochs: 60
* Batch size: 128
* Data augmentation:
  * RandomCrop
  * RandomHorizontalFlip
  * ColorJitter
  * Normalize
  * RandomErasing
* Hardware: GPU (RTX 3050TI)

## Overview of files/File structure

`network/` — model definition, training, evaluation

`data/` — dataset loading and augmentation

`test/` — evaluation and sanity-check scripts

`assets/` — diagrams and images

## Authors 

[Denys Maherovskyi](https://github.com/MaherovskyiDenys)

## Version History

* Initial Release

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Image Normalization](https://medium.com/@piyushkashyap045/image-normalization-in-pytorch-from-tensor-conversion-to-scaling-3951b6337bc8)
* [CIFAR-10 Mean-Std values](https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data)