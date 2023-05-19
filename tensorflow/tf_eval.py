import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn import metrics
from tf_cnn import *
from cnn_bayesian_hyperparam_tuning import *
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using various evaluation metrics.

    This function takes a trained model, test data (X_test and y_test), and computes
    several evaluation metrics including confusion matrix, classification report,
    and ROC curve. It provides visualizations for the confusion matrix and ROC curve.

    Args:
        model (keras.models.Sequential): Trained model.
        X_test (numpy.ndarray): Test data.
        y_test (numpy.ndarray): True labels for the test data.
    """
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Confusion matrix
    def draw_confusion_matrix(true, preds):
        conf_matx = confusion_matrix(true, preds)
        sns.heatmap(conf_matx, annot=True, annot_kws={
                    "size": 12}, fmt='g', cbar=False, cmap="viridis")
        plt.show()

    draw_confusion_matrix(y_test, y_pred)

    # Classification report
    print(classification_report(y_test, y_pred))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    display = metrics.RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Area Under Curve')
    display.plot()
    plt.show()

