from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib
import os

def train_ensemble_model(X, y):
    """
    Trains an ensemble model using Voting Classifier, including additional models like Logistic Regression.
    The voting is weighted based on F1 scores from cross-validation.
    
    Args:
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target labels (Disease Names).
    
    Returns:
    - model: Trained ensemble model.
    """
    # Encode target labels to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Define individual models
    rf = RandomForestClassifier()
    svm = SVC(probability=True)
    knn = KNeighborsClassifier()
    xgb = XGBClassifier()
    lr = LogisticRegression()

    # Train models to get their F1 scores for weighted voting
    models = {'rf': rf, 'svm': svm, 'knn': knn, 'xgb': xgb, 'lr': lr}
    f1_scores = {}
    
    for name, model in models.items():
        f1 = cross_val_score(model, X, y_encoded, cv=5, scoring='f1_macro').mean()
        f1_scores[name] = f1
    
    # Sort models by F1 score, highest first
    sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)

    # Create the ensemble with weighted voting based on F1 scores
    ensemble_model = VotingClassifier(
        estimators=[(name, models[name]) for name, _ in sorted_models],
        voting='soft',
        weights=[f1_scores[name] for name, _ in sorted_models]
    )
    
    ensemble_model.fit(X, y_encoded)
    return ensemble_model

def save_model(model, label_encoder, model_name="ensemble_model.pkl"):
    """
    Saves the trained model and label encoder to the models directory.
    
    Args:
    - model: The trained model to be saved.
    - label_encoder: The label encoder to be saved.
    - model_name (str): The name to save the model as.
    """
    # Check if the 'models' directory exists, if not create it
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, model_name)
    
    # Save the model and label encoder using joblib
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
    print(f"Model and label encoder saved to {model_path}")
