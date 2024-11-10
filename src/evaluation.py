from sklearn.metrics import accuracy_score
import joblib

# Load the trained model and label encoder
ensemble_model = joblib.load('models/ensemble_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return the accuracy.
    """
    # Encode the true labels for evaluation
    y_test_encoded = label_encoder.transform(y_test)
    
    # Get predictions from the model
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # Display the accuracy
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy
