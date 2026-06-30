## 📊 Feature Importance Visualization

**File:** `visualization/features_importance.png`  
**Description:** This plot visualizes the top 15 most influential features driving the hybrid model's churn predictions. 

### Key Insights from the Chart:
* **NLP Dominance:** Out of the top 50 features evaluated by the Random Forest model, **45 are NLP text tokens** derived from the TF-IDF matrix, while only 5 belong to the traditional tabular dataset. 
* **Linguistic Partitioning:** The chart highlights how the machine learning model heavily prioritized structural text artifacts like `recently`, `decide`, `satisfied`, and `ultimately`. 
* **Data Leakage Proof:** This visualization serves as empirical proof of the synthetic data generation bias, showing that the model achieved a 1.00 accuracy score by "memorizing" specific emotional phrasing partitions rather than discovering generalizable behavior across demographic or billing metrics.

### Code Snippet to Recreate:
The plot was generated using `seaborn.barplot` and configured to sort the Random Forest `feature_importances_` array in descending order to isolate the highest-weighted feature vectors.
