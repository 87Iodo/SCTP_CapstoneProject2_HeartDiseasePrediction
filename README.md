# Heart Disease Prediction

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning/Preparation](#data-cleaningpreparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Results/Findings](#resultsfindings)
- [Recommendations](#recommendations)
- [Limitations](#limitations)
- [References](#references)

### Project Overview

This project aims to create a predictive model in python for the segmentation of customers with heart disease. The model enables change of marketing strategy from conventional mass marketing to highly targeted marketing campaign that is more cost efficient.

### Data Sources

Patient Data: The dataset that is used to construct the model is "cardio_data.csv". It contains the required detail to understand the impact on heart. Detail information of the data can be found in "Data_Dictionary.pdf".

### Tools

- Jupyter Notebook
  - Numpy
  - Pandas
  - Matplotlib
  - Seaborn
  - Statsmodels
  - SciPy
  - scikit-learn
  - kmodes
  - kneed

### Data Cleaning/Preparation

In preparing the data, following tasks are performed:
1. Data loading
2. Inspection of data including data type, missing data, unit of each data, any spelling error from data entry for object datatype
3. Removing outliers
4. Drop irrelevant features

### Exploratory Data Analysis

EDA involves finding relation of features to heart disease:
1. Graphical visualization
2. Chi-squared test
3. Feature engineering - BMI, MAP
4. Feature creation - clustering (After splitting data to training and testing)

### Model Building and Evaluation

Different machine learning models are created and evaluated with both cross-validation and using test data. Hyperparameters optimization is performed to look for the best parameters for the model.

```
# Define the classifiers
classifiers = {
    'XGBClassifier': XGBClassifier(random_state=SEED),
    'LogisticRegression': LogisticRegression(random_state=SEED),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=SEED),
    'RandomForestClassifier': RandomForestClassifier(random_state=SEED),
    'KNeighborsClassifier': KNeighborsClassifier()
}

# Define the parameter grids for each classifier
param_grids = {
    'XGBClassifier': {'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [0.1, 0.01, 0.05],
                     'classifier__n_estimators': [100,200,300], 'classifier__min_child_weight': [1,3,5],
                     'classifier__gamma': [0.0, 0.1, 0.2], 'classifier__subsample': [0.8,0.9,1.0],
                     'classifier__colsample_bytree': [0.8,0.9,1.0]},
    'LogisticRegression': {'classifier__C': [1, 10, 20], 'classifier__solver': ['lbfgs', 'liblinear']},
    'DecisionTreeClassifier': {'classifier__max_depth': [3, 5, 25], 'classifier__max_leaf_nodes': [10,20,100]},
    'RandomForestClassifier': {'classifier__n_estimators': [100,300,500], 'classifier__max_depth': [3, 10, 50], 'classifier__max_leaf_nodes': [20,100,250]},
    'KNeighborsClassifier': {'classifier__n_neighbors': [5, 30, 50], 'classifier__weights': ['uniform','distance']}
}

# Create pipelines for each classifier
pipelines = {name: Pipeline([('classifier', clf)]) for name, clf in classifiers.items()}

# Perform grid search for each classifier
results = {}
for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    results[name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
```

### Results/Findings

The final model Decision Tree is selected based on the following criterion:
1. The performance of the model
2. The computation cost of the model
3. Interpretability of the model when presenting to stake holders.

The final model achieves 90% accuracy with 91% precision and 90% recall.
The top 3 features with highest contribution to predicting heart disease in this model are Clusters, Mean Arterial Pressure, and Age.

### Recommendations

The model's result is sufficiently high in differentiating customers who have heart disease or likely to develop heart disease. Below are recommended using the model:
- Customized advertisement for this specific group of customers
- Customized service solutions
- Customized product recommendations

### Limitations

Clustering technique used in this model needs careful adjustment based on the sample demographics. To maintain the high accuarcy, the sample data distribution used for the model should be representative of the unknown test data distribution, resulting in limited application.

### References
- [kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
