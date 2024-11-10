from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data.
    
    Args:
    - model: Trained model.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): True labels for test data.
    
    Returns:
    - dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
