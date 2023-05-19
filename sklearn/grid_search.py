### Visualizing gridsearch results requires ipython, I suggest you copy this code into a Jupyter notebook

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from plot import *


"""
    Evaluate a model by performing predictions, calculating performance metrics,
    and displaying the ROC curve and confusion matrix.

    Args:
        input_model: Model object to evaluate.
        path (str, optional): Path to save the generated plots. Defaults to None.
        tune (bool, optional): Flag indicating whether the model has already been tuned. Defaults to True.

    Returns:
        None
"""
def evaluate(input_model, X_train, y_train, X_test, y_test, path=None, tune=True):

    if not tune:
        input_model.fit(X_train, y_train)
    y_pred = input_model.predict(X_test)
    print(input_model)

    y_score1 = y_pred

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score1)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Area Under Curve')

    display.plot()
    if path is not None:
        plt.savefig(os.path.join(
            path, input_model.__class__.__name__ + 'AUC_ROC.png'))

    print(classification_report(y_test, y_pred))
    cfMat = metrics.confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    if path is not None:
        plt.savefig(os.path.join(
            path, input_model.__class__.__name__ + '_ConfusionMatrix.png'))

    return None


def Tune(input_model,X_train, y_train,param_grid, path=None):
    """
    Tune a model by performing grid search with cross-validation, displaying the best parameters,
    and generating learning curves.

    Args:
        input_model: Model object to tune.
        param_grid (dict): Parameter grid for the grid search.
        path (str, optional): Path to save the generated plots. Defaults to None.

    Returns:
        None
    """
    grid = GridSearchCV(input_model, param_grid, cv=10, refit=True, verbose=1)
    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.cv_results_['std_test_score'][grid.best_index_])

    tuned = grid
    plt.show()
    plot_grid_search(grid)
    table_grid_search(grid)
    plot_learning_curve(estimator=grid.best_estimator_, title=str(input_model) + " Learning Curve",
                        X=X_train, y=y_train, cv=10, path=path)

    evaluate(tuned, path=path, tune=True)


# Example Param space
param_grid_qdr = {
    'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'store_covariance': [True, False],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1]
}
# Parameter space for a QuadraticDiscriminantAnalysis model

# Call the functions with appropriate arguments
# evaluate(input_model, path=None, tune=True)
# Tune(input_model, param_grid, path=None)
