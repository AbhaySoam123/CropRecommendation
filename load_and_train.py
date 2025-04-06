import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib


df = pd.read_csv("/Users/abhay/CropRecommendation/Crop_recommendation.csv.xls")


X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

joblib.dump(best_model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nBest model saved:", best_model.__class__.__name__, f"with accuracy: {best_acc:.4f}")
