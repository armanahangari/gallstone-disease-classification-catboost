Gallstone Disease Classification Using CatBoost and Statistical Feature Selection

This project implements a binary classification pipeline for predicting gallstone disease status using the gallstone dataset from the UCI Machine Learning        Repository (https://archive.ics.uci.edu/dataset/1150/gallstone-1). The objective is to evaluate the effectiveness of CatBoost, combined with statistical feature  selection, in distinguishing between individuals with and without gallstone disease based on clinical and biochemical attributes.

  Data preprocessing and feature engineering
  The dataset is loaded using pandas and separated into features and the target variable (Gallstone Status). If the target variable is categorical, it is encoded     into numerical form using LabelEncoder to ensure compatibility with scikit-learn metrics and CatBoost’s training process.
  Feature selection is performed using ANOVA F-scores (f_classif), which quantify the statistical relationship between each feature and the target variable. Based    on a predefined threshold, features are divided into high-score and low-score groups. To retain potentially informative but weaker predictors, a composite          feature is constructed by averaging the low-score features, allowing the model to preserve their collective contribution without introducing excessive noise.

  The final feature set consists of:
    1) High F-score features (selected individually)
    2) A single composite feature representing low-score attributes
  The dataset is then split into training and testing subsets using a stratified train-test split to ensure reliable performance evaluation.

  Model architecture
  The classification model is built using CatBoostClassifier, a gradient boosting algorithm well-suited for structured/tabular data. The model is configured with     controlled depth, learning rate, and iteration count to balance predictive performance and generalization. CatBoost’s robustness to feature scaling and ability     to model complex nonlinear relationships make it an appropriate choice for this medical classification task.

  Evaluation metrics
  Model performance is assessed on the test set using multiple evaluation metrics:
  Accuracy, Precision, Recall, F1-score, ROC-AUC

This multi-metric evaluation provides a comprehensive view of classification performance, particularly important in medical decision-making contexts where false positives and false negatives carry different implications.

This project demonstrates a structured and interpretable approach to medical data classification, combining statistical analysis, feature engineering, and modern gradient boosting techniques in a reproducible Python workflow.
