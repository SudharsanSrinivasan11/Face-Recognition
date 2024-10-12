from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti_data = fetch_olivetti_faces()
features = olivetti_data.data
targets = olivetti_data.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)
pca = PCA(n_components=100, whiten=True)
pca.fit(X_train)
X_pca = pca.fit_transform(features)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Support Vector Machine", SVC()),
    ("Naive Bayes Classifier", GaussianNB())
]

for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, X_pca, targets, cv=kfold)
    print(f"{name}: Cross-validation Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(), param_grid, cv=5)
svm_grid.fit(X_train_pca, y_train)
best_svm = svm_grid.best_estimator_
print("Best SVM parameters:", svm_grid.best_params_)

best_svm.fit(X_train_pca, y_train)
y_pred = best_svm.predict(X_test_pca)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Test Accuracy of best SVM:", accuracy)

conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()
