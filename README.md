# Gallstone Diagnosis Using CatBoost and Feature Selection

This project implements a binary machine learning classification pipeline for diagnosing gallstone disease based on clinical and biochemical features. The dataset is sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/1150/gallstone-1) and contains patient-level attributes relevant to gallstone formation.

Data preprocessing:
The dataset is clean and well-structured, requiring minimal preprocessing. The target variable (Gallstone Status) is encoded into numerical form when necessary. Feature relevance is evaluated using ANOVA F-score analysis, and features are divided into high-importance and low-importance groups. To retain information from weaker predictors, a composite feature is constructed by aggregating low-score features, resulting in a compact and informative final feature set.

Modeling approach:
The project employs a CatBoostClassifier, a gradient boosting algorithm optimized for tabular data, to model complex nonlinear relationships without extensive feature scaling or transformation. The model is trained using a reproducible train-test split and configured to balance predictive performance and generalization.

Evaluation:
Model performance is assessed using multiple metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, providing a comprehensive evaluation suitable for medical classification tasks.
