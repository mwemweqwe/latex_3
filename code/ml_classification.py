import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score

# Генерация синтетических данных
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_classes=2, random_state=42)

# Метод опорных векторов
svm_model = svm.SVC(kernel='rbf', C=1.0, random_state=42)
svm_scores = cross_val_score(svm_model, X, y, cv=5)
print(f"SVM accuracy: {svm_scores.mean():.3f}")

# Логистическая регрессия
lr_model = LogisticRegression(random_state=42)
lr_scores = cross_val_score(lr_model, X, y, cv=5)
print(f"Logistic Regression accuracy: {lr_scores.mean():.3f}")

# Градиентный бустинг
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_scores = cross_val_score(gb_model, X, y, cv=5)
print(f"Gradient Boosting accuracy: {gb_scores.mean():.3f}")