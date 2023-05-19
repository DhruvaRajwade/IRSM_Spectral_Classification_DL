import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_roc_curve
from matplotlib import pyplot


def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a classifier on the given training and test data.

    Args:
        classifier: The classifier instance to evaluate.
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        y_train (array-like): Target values for the training data.
        X_test (array-like): Test data matrix of shape (n_samples, n_features).
        y_test (array-like): Target values for the test data.

    Returns:
        None
    """
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    print("Accuracy:", '{0:.2%}'.format(accuracy_score(y_test, prediction)))
    print("Cross Validation Score:", '{0:.2%}'.format(cross_val_score(
        classifier, X_train, y_train, cv=cv, scoring='roc_auc').mean()))
    print("ROC_AUC Score:", '{0:.2%}'.format(
        roc_auc_score(y_test, prediction)))
    plot_roc_curve(classifier, X_test, y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()


def evaluate_model_classification(classifier, X_test, y_test, colors=['#F93822', '#FDD20E']
):
    """
    Evaluate a classifier on the given test data and display the confusion matrix and classification report.

    Args:
        classifier: The classifier instance to evaluate.
        X_test (array-like): Test data matrix of shape (n_samples, n_features).
        y_test (array-like): Target values for the test data.

    Returns:
        None
    """
    # Confusion Matrix
    cm = confusion_matrix(y_test, classifier.predict(X_test))
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value)
                   for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2,
              v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, cmap=colors, fmt='')

    # Classification Report
    print(classification_report(y_test, classifier.predict(X_test)))


def run_functions_on_models(estimators, function1, function2, X_train, y_train, X_test, y_test):
    """
    Run the given functions on every single classifier instance in scikit-learn.

    Args:
        estimators: List of (name, estimator_class) tuples representing the available classifiers.
        function1: The first function to run on each classifier.
        function2: The second function to run on each classifier.
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        y_train (array-like): Target values for the training data.
        X_test (array-like): Test data matrix of shape (n_samples, n_features).
        y_test (array-like): Target values for the test data.

    Returns:
        None
    """
    for name, estimator_class in estimators:
        try:
            estimator = estimator_class()
            print(
                f"Running {function1.__name__} and {function2.__name__} on {name}...")
            estimator.fit(X_train, y_train)
            function1(estimator, X_train, y_train, X_test, y_test)
            function2(estimator, X_test, y_test)
            print("Completed successfully!")
        except Exception as e:
            print(f"Error encountered while fitting {name}: {str(e)}")
            continue


def visualize_best_models(estimators, X_train, y_train):
    """
    Visualize the performances of the best models using violin plots, box plots, and bar plots.

    Args:
        estimators: List of (name, estimator_class) tuples representing the available classifiers.
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        y_train (array-like): Target values for the training data.

    Returns:
        None
    """
    # Dictionary to store the cross-validation scores of all models
    scores = {}

    # Iterate through the list of estimator classes
    for name, ClassifierClass in estimators:
        try:
            # Initialize the classifier
            clf = ClassifierClass()
            # Perform cross-validation to evaluate the performance of the classifier
            cv_scores = cross_val_score(
                clf, X_train, y_train, cv=10, scoring='roc_auc')
            # Store the cross-validation scores of the classifier in the dictionary
            scores[name] = cv_scores
        except:
            # Skip any classifiers that raise errors
            pass

    # Select the 5 models with the highest average performance
    best_models = sorted(scores.keys(), key=lambda x: -scores[x].mean())[:5]

    # Create violin plots of the performances of the 5 best models
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[scores[name] for name in best_models])
    plt.xticks(ticks=range(5), labels=[f"Model {i+1}" for i in range(5)])
    plt.xlabel('Model')
    plt.ylabel('AUC_ROC Scores')
    plt.title('Performances of the Best Models')
    plt.legend(best_models, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Create box plots of the performances of the 5 best models
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[scores[name] for name in best_models])
    plt.xticks(ticks=range(5), labels=[f"Model {i+1}" for i in range(5)])
    plt.xlabel('Model')
    plt.ylabel('AUC_ROC Scores')
    plt.title('Performances of the Best Models')
    plt.legend(best_models, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Create bar plots of the mean performances of the 5 best models
    plt.figure(figsize=(5, 5))
    sns.barplot(x=[f"Model {i+1}" for i in range(5)],
                y=[scores[name].mean() for name in best_models])
    plt.xlabel('Model')
    plt.ylabel('Mean AUC_ROC Scores')
    plt.title('Mean Performances of the Best Models')
    plt.ylim(0.5, 1.0)
    plt.legend(best_models, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def plot_roc_curves_best_models(estimators, X_train, y_train, X_test, y_test):
    """
    Plot the ROC curves for the best models.

    Args:
        estimators: List of (name, estimator_class) tuples representing the available classifiers.
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        y_train (array-like): Target values for the training data.
        X_test (array-like): Test data matrix of shape (n_samples, n_features).
        y_test (array-like): Target values for the test data.

    Returns:
        None
    """
    # Initialize a set to keep track of unique classifiers
    unique_classifiers = set()

    # Iterate through the list of estimator classes
    for name, ClassifierClass in estimators:
        try:
            # Initialize the classifier
            clf = ClassifierClass()
            # Fit the classifier to the training data
            clf.fit(X_train, y_train)
            # Predict the test data
            y_pred = clf.predict(X_test)
            # Calculate the ROC AUC score
            auc = roc_auc_score(y_test, y_pred)
            if auc > 0.85:
                # Calculate the false positive rate and true positive rate for the ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                # Check if the classifier has already been plotted
                if name not in unique_classifiers:
                    # Plot the ROC curve for the classifier
                    plt.plot(fpr, tpr, marker='.',
                             label=f"{name} (AUC = {auc:.2f})")
                    # Add the classifier to the set of unique classifiers
                    unique_classifiers.add(name)
        except:
            # Skip any classifiers that raise errors
            pass

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for the Best Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


# Run the functions on every single classifier instance in the SkLearn library
"""
run_functions_on_models(estimators, model, model_evaluation)
visualize_best_models(estimators, X_train, y_train)
plot_roc_curves_best_models(estimators, X_train, y_train, X_test, y_test)

"""