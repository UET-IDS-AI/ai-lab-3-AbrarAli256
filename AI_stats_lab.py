"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    top_3_feature_indices = list(np.argsort(np.abs(model.coef_))[-3:])
    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices

    raise NotImplementedError


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """
    X, y = load_diabetes(return_X_y=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    mean_r2 = scores.mean()
    std_r2 = scores.std()
    
    return mean_r2, std_r2

    raise NotImplementedError


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)

    cm = confusion_matrix(y_test, test_preds)
    
    """False Negative (FN) medically means: the model predicted the patient does NOT have cancer,
    but they actually DO. This is the most dangerous error in medical diagnosis — a patient
    with cancer is told they are healthy and receives no treatment, allowing the disease to progress.
    This is why recall (sensitivity) is critical in medical models — we want to minimize FNs."""

    return train_accuracy, test_accuracy, precision, recall, f1

    raise NotImplementedError


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """
    # When C is very SMALL (e.g. 0.01):
    #   Regularization is STRONG → model penalized for large coefficients
    #   → simpler model → may UNDERFIT → lower train and test accuracy
    #
    # When C is very LARGE (e.g. 100):
    #   Regularization is WEAK → model free to grow large coefficients
    #   → complex model → may OVERFIT → high train accuracy, lower test accuracy
    #
    # OVERFITTING happens at large C values.
    # C = 1 / regularization_strength

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results_dictionary = {}

    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(max_iter=5000, C=C, random_state=42)
        model.fit(X_train, y_train)

        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))

        results_dictionary[C] = (train_accuracy, test_accuracy)

    return results_dictionary

# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy
    """
    # Cross-validation is especially important in medical diagnosis because:
    #
    # 1. HIGH STAKES: A single train/test split might get "lucky" or "unlucky"
    #    with how patients are divided. In medicine, we need RELIABLE estimates
    #    of model performance — not a one-time result.
    #
    # 2. LIMITED DATA: Medical datasets are often small (privacy, cost of collection).
    #    CV makes full use of all data for both training and validation rather than
    #    wasting a chunk as a fixed test set.
    #
    # 3. GENERALIZATION ASSURANCE: We need confidence the model works on UNSEEN patients
    #    from different batches/hospitals. CV tests this across multiple folds,
    #    and std_accuracy tells us how stable the model is across different patient groups.
    #
    # 4. AVOID OVERFITTING BLINDNESS: A single split may not catch a model that
    #    memorized training data. CV exposes this by testing on 5 different subsets.

    X, y = load_breast_cancer(return_X_y=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    return mean_accuracy, std_accuracy
"""
The `std_accuracy` is particularly meaningful here — it tells you **how consistent** the model is:
Low std  → model performs similarly on all patient groups → trustworthy
High std → model is sensitive to which patients it sees   → unreliable
"""

    raise NotImplementedError
