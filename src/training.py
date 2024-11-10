from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_ensemble_model(X, y):
    """
    Trains an ensemble model using Voting Classifier, including additional models like Logistic Regression.
    
    Args:
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target labels.
    
    Returns:
    - model: Trained ensemble model.
    """
    rf = RandomForestClassifier()
    svm = SVC(probability=True)
    knn = KNeighborsClassifier()
    xgb = XGBClassifier()
    lr = LogisticRegression()

    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('knn', knn), ('xgb', xgb), ('lr', lr)],
        voting='soft'
    )

    ensemble_model.fit(X, y)
    return ensemble_model

def save_model(model, model_name="ensemble_model.pkl"):
    """
    Saves the trained model to the models directory in the GitHub repo.
    
    Args:
    - model: The trained model to be saved.
    - model_name (str): The name to save the model as. Defaults to 'ensemble_model.pkl'.
    """
    # Check if the 'models' directory exists, if not create it
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, model_name)
    
    # Save the model using joblib
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
