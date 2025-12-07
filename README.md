Project: Adversarial Robustness on FER2013

1. Overview

This project trains a CNN on the FER2013 facial expression dataset and evaluates its resistance to adversarial manipulation. The work includes two attack methods, two defense methods and a validation step using Foolbox to confirm correctness.

Implemented attacks
PGD (Projected Gradient Descent) under L infinity
DeepFool under L2

Implemented defenses
Feature Squeezing with spatial smoothing
Adversarial training using PGD and DeepFool examples

2. Dataset

Primary dataset: FER2013
Source: https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge

Dataset file: fer2013.csv
Image size: 48x48 grayscale faces

Columns in the file
pixels: space-separated pixel values
emotion: integer label in the range 0 to 6
Usage: defines the train, validation and test split

Dataset split
Training: Usage equals "Training" (28,709 images)
Validation: Usage equals "PublicTest" (3,589 images)
Test: Usage equals "PrivateTest" (3,589 images)

Total images: 35,887

3. Environment and Dependencies

Recommended environment
Python 3.8 or newer with GPU acceleration

Main packages used
numpy
pandas
matplotlib
scikit-learn
tensorflow 2.x with tf.keras
foolbox version 3.3.4

Installation example
pip install numpy pandas matplotlib scikit-learn tensorflow
pip install foolbox==3.3.4

4. Models and Saved Files

baseline_cnn.h5
Model trained only on clean FER2013 data.

robust_cnn_adv_pgd_df_pro.h5
Model further trained using adversarial examples from PGD and DeepFool.

If these files are present, the notebook loads them automatically to avoid retraining.

5. How To Run

Step 1  
Open the notebook (ipynb) in Jupyter, Colab or Kaggle and enable GPU if available.

Step 2  
Ensure fer2013.csv is accessible and that the path inside the notebook is correct.

Step 3  
Execute all notebook cells from top to bottom:
Data loading and preprocessing
Baseline CNN training or loading
PGD and DeepFool attack generation
Evaluation on clean and adversarial examples
Feature Squeezing defense evaluation
Adversarial training and evaluation of the robust model
Foolbox-based validation of attack implementations

Step 4  
Review printed metrics and plots to compare performance across models and attack or defense configurations.
