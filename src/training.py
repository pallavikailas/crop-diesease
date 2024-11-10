from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_ensemble_model(X, y):
    """
    Trains an ensemble model using Voting Classifier.
    
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

    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('knn', knn), ('xgb', xgb)],
        voting='soft'
    )

    ensemble_model.fit(X, y)
    return ensemble_model

def load_model(model_path):
    return joblib.load(model_path)

def save_model(model, model_path):
    joblib.dump(model, model_path)
