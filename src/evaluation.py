from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model and label encoder
ensemble_model = joblib.load('models/ensemble_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return performance metrics.
    """
    # Encode the true labels for evaluation
    y_test_encoded = label_encoder.transform(y_test)
    
    # Get predictions from the model
    y_pred = model.predict(X_test)
    
    # Decode the predicted labels back to string labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_decoded)
    precision = precision_score(y_test, y_pred_decoded, average="macro")
    recall = recall_score(y_test, y_pred_decoded, average="macro")
    f1 = f1_score(y_test, y_pred_decoded, average="macro")
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    return metrics
