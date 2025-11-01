# The Lottery Ticket Hypothesis: A Mini Research Study on CIFAR-10

## Authors

This mini research study was developed by Johan Nielsen, Lasse Abildhauge Christensen and Jakob Ahlberg as part of Assignment 3, investigating the
Lottery Ticket Hypothesis on convolutional neural networks.

## Background

This study extends the work of Frankle and Carbin in "The Lottery Ticket Hypothesis:
Finding Small, Trainable Neural Networks" (https://arxiv.org/abs/1803.03635). Their
paper explores why large, overparameterized networks are easier to train than smaller
networks found through pruning. Their answer is the lottery ticket hypothesis:

> Any large network that trains successfully contains a subnetwork that is
> initialized such that - when trained in isolation - it can match the
> accuracy of the original network in at most the same number of training
> iterations.

They refer to this special subset as a *winning ticket*.

Frankle and Carbin conjecture that pruning a neural network after training reveals a
winning ticket in the original, untrained network. They posit that weights that were
pruned after training were never necessary at all, meaning they could have been removed
from the original network with no harm to learning. Once pruned, the original network
becomes a winning ticket.

## Purpose

This study implements and evaluates the lottery ticket hypothesis on a convolutional
neural network for CIFAR-10 classification. Unlike the original work which focused on
fully-connected networks for MNIST, this research investigates whether the hypothesis
holds for more complex architectures and datasets. The experiment uses an iterative
pruning strategy that prunes weights based on the difference between best-performing
and final epoch weights.

## Methodology

To evaluate the lottery ticket hypothesis in the context of pruning, we run the
following experiment:

1. Randomly initialize a convolutional neural network.

2. Train the network until it converges (26 epochs), keeping track of both the best
   validation loss weights and final epoch weights.

3. Prune a fraction (10%) of the network weights based on a global threshold computed
   from the absolute difference between best-model and final-model weights. Weights
   with larger differences are considered less stable and are pruned first.

4. To extract the winning ticket, reset the weights of the remaining portion of the
   network to their values from (1) - the initializations they received before training
   began.

5. To evaluate whether the resulting network at step (4) is indeed a winning ticket,
   train the pruned network and examine its convergence behavior and accuracy.

6. Repeat steps 2-5 iteratively (15 times) to progressively prune the network while
   maintaining performance.

Our pruning strategy differs from the original paper by using weight stability (the
difference between best and final weights) as the criterion for pruning, rather than
magnitude-based pruning. This approach tests whether unstable weights that change
significantly during training are less critical for performance.

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

### Two-Model Approach

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

### Evaluation Metrics

The study tracks several metrics throughout the experiment:
- **Test Accuracy**: Performance on the held-out test set
- **Test Loss**: Categorical crossentropy on test set
- **Sparsity**: Percentage of weights that are exactly zero
- **Validation Loss**: Used for early stopping and model selection
- **Validation Accuracy**: Secondary metric for model selection

## Results and Observations

The experiment demonstrates the following:

1. **Initial Performance**: The dense (unpruned) model achieves ~74.75% test accuracy.

2. **Gradual Degradation**: As sparsity increases from 0% to ~77%, test accuracy
   gradually decreases. Early pruning iterations (up to ~40% sparsity) maintain
   relatively stable performance near the original accuracy.

3. **Critical Threshold**: Beyond approximately 65% sparsity, the model performance
   degrades significantly, eventually collapsing to random chance (10% accuracy for
   CIFAR-10's 10 classes).

4. **Weight Stability Hypothesis**: The weight-stability-based pruning strategy shows
   that weights with large differences between best and final models can indeed be
   removed without immediate catastrophic performance loss, supporting the notion that
   these weights may be less critical for the learned representation.

5. **Iterative Pruning Benefits**: The iterative approach allows the model to adapt
   to pruning over multiple training cycles, potentially maintaining better performance
   than one-shot pruning to high sparsity levels.

## Code Structure

The experiment is implemented in a Jupyter notebook (`Assignment3_code.ipynb`) with the
following main sections:

1. **Data Preparation**: Downloads and preprocesses CIFAR-10 data with proper train/val/test splits.

2. **Masking Utilities**: Implements functions and callbacks to maintain weight masks
   and prevent pruned weights from regrowing.

3. **Model Definition**: Defines the convolutional neural network architecture using
   Keras Sequential API.

4. **Initial Training**: Trains the full model to convergence, tracking both best and
   final weights.

5. **Pruning Function**: Implements global mask-aware pruning based on weight stability
   (difference between best and final weights).

6. **Iterative Retraining Loop**: Performs 15 iterations of:
   - Pruning 10% of active weights
   - Retraining the pruned network with masks applied
   - Evaluating on test set
   - Tracking metrics and sparsity

7. **Visualization**: Generates plots showing:
   - Training/validation accuracy and loss curves for each iteration
   - Best validation loss and accuracy across iterations
   - Validation loss per epoch for all training runs
   - Test accuracy, loss, and sparsity across all pruning levels

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

## Experimental Findings

The study investigates several key questions:

1. **Does the lottery ticket hypothesis hold for CNNs on CIFAR-10?**
   - Results show that sparse subnetworks can maintain reasonable performance up to
     moderate sparsity levels (~40-50%), but performance degrades significantly beyond
     that point.

2. **Is weight stability a useful pruning criterion?**
   - The weight-difference-based pruning strategy successfully identifies weights that
     can be removed without immediate catastrophic loss, suggesting that unstable
     weights may indeed be less critical.

3. **How does iterative pruning compare to one-shot pruning?**
   - The iterative approach allows gradual adaptation, potentially maintaining better
     performance than aggressive one-shot pruning.

4. **At what sparsity level does the winning ticket break down?**
   - The experiment shows that beyond ~65% sparsity, the model performance collapses,
     indicating a critical threshold for this architecture and task.

## Limitations and Future Work

This mini research study has several limitations:

- Single architecture and dataset: Results may not generalize to other architectures
  or datasets.
- Fixed pruning fraction: All iterations use 10% pruning, not exploring adaptive
  pruning strategies.
- Limited trials: Only one experimental run, not demonstrating statistical
  significance.
- Pruning criterion: Weight stability may not be optimal; comparison with magnitude-based
  pruning would be valuable.

Potential future directions include:
- Testing multiple architectures and datasets
- Comparing different pruning criteria (magnitude, gradient, etc.)
- Running multiple trials to establish statistical significance
- Exploring adaptive pruning schedules
- Investigating whether winning tickets exist at initialization (the core claim of
  the original hypothesis)

*This is a mini research study developed for educational purposes.*
