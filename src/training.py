from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(X, y, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC()
    elif model_type == "knn":
        model = KNeighborsClassifier()
    elif model_type == "xgboost":
        model = XGBClassifier()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X, y)
    return model

def load_model(model_path):
    return joblib.load(model_path)

def save_model(model, model_path):
    joblib.dump(model, model_path)
