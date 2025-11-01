# The Lottery Ticket Hypothesis: A Mini Research Study on CIFAR-10

## Authors

This mini research study was developed by Johan Nielsen, Lasse Abildhauge Christensen and Jakob Ahlberg as part of Assignment 3, investigating the
Lottery Ticket Hypothesis on convolutional neural networks.


## Dataset and Architecture

**Dataset**: CIFAR-10 (50,000 training images, 10,000 test images)
- Training set: 45,000 samples
- Validation set: 5,000 samples
- Test set: 10,000 samples
- Image size: 32x32x3 (RGB)
- 10 classes

**Model Architecture**:
- Conv2D(64, 3x3) → ReLU
- Conv2D(64, 3x3) → ReLU
- MaxPooling2D(2x2)
- Conv2D(128, 3x3) → ReLU
- Conv2D(128, 3x3) → ReLU
- MaxPooling2D(2x2)
- Flatten
- Dense(256) → ReLU
- Dense(256) → ReLU
- Dense(10) → Softmax

**Training Hyperparameters**:
- Optimizer: Adam (learning rate: 3e-4)
- Loss: Categorical crossentropy
- Batch size: 60
- Initial training epochs: 26
- Retraining epochs per iteration: 25
- Pruning fraction per iteration: 10%
- Number of pruning iterations: 15

## Key Components

### Masking System

The codebase implements a masking system to prevent pruned weights from regrowing
during retraining. After each pruning step, masks are created that permanently
disable pruned weights. During retraining, these masks are applied after each batch
update to ensure pruned weights remain zero.

### Hight-Movement Pruning

The study maintains two versions of the model:
- **Best Model**: Contains weights from the epoch with lowest validation loss
- **Final Model**: Contains weights from the final training epoch

The pruning strategy uses the difference between these two models (`|w_best - w_final|`)
to identify which weights are least stable and therefore candidates for pruning.

### Iterative Pruning

The experiment performs 15 iterations of pruning and retraining:
- Each iteration prunes 10% of the currently active weights
- Pruning uses a global quantile threshold across all layers (except the output layer)
- Active weights are those not already pruned in previous iterations
- If best and final weights converge (become identical), pruning is skipped for that
  iteration

## Getting Started

1. Install required dependencies:
   ```bash
   pip install tensorflow keras numpy scikit-learn matplotlib carbontracker
   ```

2. Open and run `Assignment3_code.ipynb` in a Jupyter notebook environment (e.g., Google Colab).

3. The notebook will automatically:
   - Download CIFAR-10 data
   - Train the initial model
   - Perform iterative pruning and retraining
   - Generate visualizations of results

