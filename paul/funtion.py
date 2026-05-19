from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
 
def evaluate_model(name, y_test, y_pred, classes=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred,
                           average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred,
                        average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred,
                    average='weighted', zero_division=0)
 
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted avg)")
    print(f"  Recall    : {rec:.4f}  (weighted avg)")
    print(f"  F1 Score  : {f1:.4f}  (weighted avg)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
 
    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(name.lower().replace(' ','_') + '_cm.png', dpi=150)
    plt.show()
 
    return {'model': name, 'accuracy': acc,
            'precision': prec, 'recall': rec, 'f1': f1}
