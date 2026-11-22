# Gallstone Disease Prediction Using CatBoost and Composite Feature Optimization

This project predicts gallstone disease from clinical data, including bioelectrical impedance measurements and routine laboratory results. The data comes from the public [UCI Gallstone dataset](https://archive.ics.uci.edu/dataset/1150/gallstone-1), which contains 319 patients with 38 features collected at a hospital in Ankara. The aim is to evaluate whether keeping the most informative features and merging the weaker ones into a single composite feature improves a gradient boosting classifier.

- Data preprocessing and feature engineering: The dataset is clean and balanced, so no further preprocessing was needed. All features are ranked with ANOVA F-scores, and a threshold of 0.9 is applied. Features above it are kept individually, and the lower-scoring ones are averaged into a single composite feature, `LowScore_Composite`.
- Modeling approach: A CatBoost classifier is trained on the resulting feature set with 500 iterations, a learning rate of 0.05, a maximum depth of 6, and the Logloss objective with AUC as the evaluation metric. The data is split 75/25 into training and test sets with a fixed random seed.
- Evaluation: On the held-out test set the model reaches an accuracy of 88.75%, a precision of 90.00%, a recall of 87.80%, an F1-score of 88.89%, and an AUC of 93.81%.

## Paper

Arman Ahangari Kiasari, "Improving Gallstone Disease Prediction through Composite Feature Optimization with CatBoost," 2nd International Biennial Conference on Advances in Artificial Intelligence and Data Science (IBAAIDS), IEEE, 2026. [doi:10.1109/IBAAIDS66771.2026.11567395](https://doi.org/10.1109/IBAAIDS66771.2026.11567395)
