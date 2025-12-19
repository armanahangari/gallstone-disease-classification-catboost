import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

df = pd.read_csv('gallstone.csv')
X = df.drop(columns=['Gallstone Status'])
y = df['Gallstone Status']

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

f_scores, p_values = f_classif(X, y)
f_score_df = pd.DataFrame({
    'Feature': X.columns,
    'F_score': f_scores,
    'p_value': p_values
})

low_high = 5
low_score_features = f_score_df[f_score_df['F_score'] < low_high]['Feature'].values
high_score_features = f_score_df[f_score_df['F_score'] >= low_high]['Feature'].values

df['LowScore_Composite'] = X[low_score_features].mean(axis=1)

X_final = pd.concat([X[high_score_features], df[['LowScore_Composite']]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42)
model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
