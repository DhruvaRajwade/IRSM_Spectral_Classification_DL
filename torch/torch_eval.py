from torch import nn
from torch_models import *
from torch_train import *
from vae_loss import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

def eval_model(model,loss_fn, X_test_latent, y_test, plot_auc=False, plot_report=False, plot_confusion=False):
    """
    Evaluates a PyTorch model on the test data and prints out the accuracy, classification report and confusion matrix.

    Args:
    - model (torch.nn.Module): The trained PyTorch model.
    - X_test_latent (torch.Tensor): The final latent representation of the test data.
    - y_test (torch.Tensor): The true labels of the test data.
    - plot_auc (bool): Whether to plot the ROC curve and AUC score. Default is False.
    - plot_report (bool): Whether to print the classification report. Default is False.
    - plot_confusion (bool): Whether to plot the confusion matrix. Default is False.
    """

    # Test the model on the test data and evaluate its performance
    y_pred = model(X_test_latent)
    loss = loss_fn(y_pred, y_test)
    print(f'Test loss: {loss.item()}')

    # Convert the predicted probabilities to class labels
    y_pred = y_pred.argmax(axis=1)
    y_test2 = y_test.argmax(axis=1)

    # Calculate the accuracy
    accuracy = (y_pred == y_test2).float().mean()
    print(f'Accuracy: {accuracy.item()}')

    # Plot ROC curve and calculate AUC score if requested
    if plot_auc:
        fpr, tpr, thresholds = metrics.roc_curve(y_test2, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Area Under Curve')
        display.plot()
        plt.show()

    # Print classification report if requested
    if plot_report:
        report = classification_report(y_test2, y_pred, target_names=['class 0', 'class 1'])
        print(report)

    # Plot confusion matrix if requested
    if plot_confusion:
        conf_matx = confusion_matrix(y_test2, y_pred)
        sns.heatmap(conf_matx, annot=True, annot_kws={"size": 12}, fmt='g', cbar=False, cmap="viridis")
        plt.show()