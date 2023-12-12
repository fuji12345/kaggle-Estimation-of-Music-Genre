import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from hydra.utils import to_absolute_path
from sklearn.metrics import confusion_matrix


def plot_feature_Importance(model, i_fold, columns):
    feature_importances = model.model.feature_importances_

    path = Path(to_absolute_path(f"outputs/feature_importances/result{i_fold}"))
    path.parent.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10, 6))
    plt.barh(columns, feature_importances)
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importance")
    plt.savefig(path)


def plot_metrix(i_fold, val_y, val_predict):
    path = Path(to_absolute_path(f"outputs/metrix/result{i_fold}"))
    path.parent.mkdir(exist_ok=True, parents=True)

    cm = confusion_matrix(val_y, val_predict)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(to_absolute_path(path))
