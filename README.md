Project: Adversarial Robustness on FER2013

## 1. Overview

This project trains a CNN on the FER2013 facial expression dataset and
studies how robust it is to adversarial attacks.

Implemented attacks: • PGD (Projected Gradient Descent) under L infinity
• DeepFool under L2

Implemented defenses: • Feature Squeezing with spatial smoothing •
Adversarial training using PGD and DeepFool examples

We also use Foolbox to validate the custom attack implementations.

## 2. Environment and Dependencies

Recommended: Python 3.8+ with GPU support.

Main packages: • numpy, pandas, matplotlib • scikit-learn • tensorflow
(2.x) with tf.keras • foolbox==3.3.4

Example installation:

pip install numpy pandas matplotlib scikit-learn tensorflow pip install
foolbox==3.3.4

## 3. Dataset: FER2013 and Splits

Input file: fer2013.csv (48x48 grayscale faces).

Columns: • "pixels": space separated pixel values • "emotion": integer
label in \[0, 6\] • "Usage": dataset split indicator

The split is done using the "Usage" column: • Training: Usage ==
"Training" 28,709 images • Validation: Usage == "PublicTest" 3,589
images • Test: Usage == "PrivateTest" 3,589 images

Total: 35,887 images (about 80 percent / 10 percent / 10 percent).

## 4. Models and Saved Files

The notebook saves and loads:

• baseline_cnn.h5\
CNN trained only on clean data.

• robust_cnn_adv_pgd_df_pro.h5\
CNN further trained with adversarial examples (PGD + DeepFool).

If these files exist, the notebook will reuse them instead of
retraining.

## 5. How To Run

1)  Open the notebook (ipynb) in Jupyter, Colab or Kaggle.\
    Enable GPU if possible.

2)  Make sure fer2013.csv is available and the path in the notebook
    points to it.

3)  Run all cells from top to bottom: • Data loading and preprocessing
    (including dataset split) • Baseline CNN definition and training or
    loading • PGD and DeepFool attack generation • Evaluation on clean
    and adversarial examples • Feature Squeezing defense • Adversarial
    training and evaluation of the robust model • Foolbox based
    validation

4)  Inspect printed accuracies and plots to compare: • Baseline vs
    attacks • Baseline + Feature Squeezing • Adversarially trained model
