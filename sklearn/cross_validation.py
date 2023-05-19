import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from matplotlib import pyplot as plt


def evaluate_model(cv, model, X, y):
    """
    Evaluate the performance of a model using cross-validation.

    Args:
        cv: Cross-validation strategy object.
        model: Model object to evaluate.
        X (array-like): Input data matrix of shape (n_samples, n_features).
        y (array-like): Target values of shape (n_samples,).

    Returns:
        mean_score: Mean cross-validation score.
    """
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    mean_score = np.mean(scores)
    return mean_score


def sensitivity_analysis(model_list, X_train, y_train):
    """
    Perform a sensitivity analysis of the number of folds used in K-fold cross-validation.

    Args:
        model_list: List of models to analyze.
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        y_train (array-like): Target values for the training data.

    Returns:
        None
    """
    r_max = X_train.shape[1]
    folds = range(2, r_max)
    means, mins, maxs = [], [], []

    for model in model_list:
        ideal, _, _ = evaluate_model(LeaveOneOut(), model, X_train, y_train)
        print(type(model).__name__)
        print('Ideal: %.3f' % ideal)

        for k in folds:
            cv = KFold(n_splits=k, shuffle=True, random_state=1)
            k_mean = evaluate_model(cv, model, X_train, y_train)
            k_min = k_mean - np.min(k_mean)
            k_max = np.max(k_mean) - k_mean
            print('> folds=%d, accuracy=%.3f (%.3f, %.3f)' %
                  (k, k_mean, k_min, k_max))
            means.append(k_mean)
            mins.append(k_min)
            maxs.append(k_max)

        plt.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
        plt.plot(folds, [ideal for _ in range(len(folds))], color='r')
        plt.xlabel('Number of Folds')
        plt.ylabel('Accuracy')
        plt.title('Sensitivity Analysis of K-Fold Cross-Validation')
        plt.show()


def pearson_correlation_analysis(model_list, X_train, y_train):
    """
    Perform Pearson correlation analysis between ideal cross-validation and K-fold cross-validation.

    Args:
        model_list: List of models to analyze.
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        y_train (array-like): Target values for the training data.

    Returns:
        None
    """
    ideal_cv = LeaveOneOut()
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    ideal_results, cv_results = [], []

    for model in model_list:
        cv_mean = evaluate_model(cv, model, X_train, y_train)
        ideal_mean = evaluate_model(ideal_cv, model, X_train, y_train)

        if np.isnan(cv_mean) or np.isnan(ideal_mean):
            continue

        cv_results.append(cv_mean)
        ideal_results.append(ideal_mean)

        print('>%s: ideal=%.3f, cv=%.3f' %
              (type(model).__name__, ideal_mean, cv_mean))

    corr, _ = pearsonr(cv_results, ideal_results)
    print('Correlation: %.3f' % corr)

    plt.scatter(cv_results, ideal_results)
    coeff, bias = np.polyfit(cv_results, ideal_results, 1)
    line = coeff * np.asarray(cv_results) + bias
    plt.plot(cv_results, line, color='r')
    plt.xlabel('K-Fold Cross-Validation Accuracy')
    plt.ylabel('Ideal Cross-Validation Accuracy')
    plt.title('Pearson Correlation Analysis')
    plt.show()

#Example usage
"""
sensitivity_analysis(model_list, X_train, y_train)
pearson_correlation_analysis(model_list, X_train, y_train)

"""