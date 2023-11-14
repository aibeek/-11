from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  # Add this import

# Assuming you have a dataset for training and testing
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset

# Placeholder for XGBoost model training step
# Replace this with your actual training code
params = {'objective': 'binary:logistic', 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'alpha': 10}
num_round = 100
dtrain = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train(params, dtrain, num_round)

# Анализ важности признаков
feature_importance = bst.get_score(importance_type='weight')
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("Feature Importance:")
for feature, importance in feature_importance:
    print(f"{feature}: {importance}")

# Оптимизация параметров модели
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}
grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1, alpha=10),
                           param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Обучаем модель с оптимальными параметрами
bst_optimized = xgb.train({**params, **best_params}, dtrain, num_round)

# Прогнозирование на тестовой выборке с оптимизированной моделью
y_pred_optimized = bst_optimized.predict(xgb.DMatrix(X_test))
y_pred_optimized_binary = [1 if p >= 0.5 else 0 for p in y_pred_optimized]

# Оценка качества оптимизированной модели
accuracy_optimized = accuracy_score(y_test, y_pred_optimized_binary)
print(f'Accuracy (Optimized): {accuracy_optimized}')

# Визуализация ROC-кривой
fpr, tpr, _ = roc_curve(y_test, y_pred_optimized)
roc_auc = roc_auc_score(y_test, y_pred_optimized)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred_optimized_binary)
print("Confusion Matrix:")
print(conf_matrix)
