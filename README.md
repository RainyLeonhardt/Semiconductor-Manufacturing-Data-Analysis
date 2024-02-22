# Semiconductor Manufacturing Data Analysis

## Overview
This project analyzes a semiconductor manufacturing dataset (`uci-secom.csv`) from Kaggle to extract insights and identify potential applications. It employs various data science methodologies, including data preprocessing, machine learning, and data visualization.

## Dataset Description
The dataset includes:
- A `Time` column with the timestamp of data collection.
- 590 feature columns (`Sensor_0` to `Sensor_589`) representing sensor data or process measurements.
- A `Pass/Fail` label column indicating the manufacturing outcome (1 for fail, -1 for pass).

## Project Goals
- To understand the data structure, quality, and distribution.
- To address missing values and imbalanced data.
- To uncover patterns and insights through data visualization.
- To predict the Pass/Fail outcome using machine learning models.
- To evaluate the performance of preprocessing techniques and models.

## Data Preprocessing
1. **Handling Missing Values:** Explored deletion, mean/median imputation, and hybrid approaches.
2. **Addressing Imbalanced Data:** Applied undersampling and oversampling (SMOTE) techniques.
3. **Normalization:** Utilized Min-Max scaling to prepare the dataset for machine learning algorithms.

## Data Analysis and Visualization
- Conducted exploratory data analysis (EDA) using box plots and bar charts.
- Analyzed the correlation between features and the Pass/Fail outcome.

## Machine Learning
- **Baseline Model:** Implemented XGBoost to handle imbalanced data, evaluating its performance with precision, recall, F1 score, and confusion matrices.
- **Model Comparison:** Compared different preprocessing approaches to optimize handling missing values and imbalanced data.
- **Advanced Techniques:** Explored dimensionality reduction (PCA) and custom oversampling techniques.

## Key Findings
- Preprocessing significantly impacts handling missing values and imbalanced data, with certain strategies proving more effective.
- Machine learning models, particularly those adjusted for imbalanced data, can predict the Pass/Fail outcome effectively.
- Novel oversampling techniques, inspired by CTGAN and SMOTE, improved model recall without significantly affecting precision.

## Future Directions
- Experiment with other machine learning algorithms and parameter tuning.
- Explore additional oversampling and undersampling techniques.
- Validate the model on different datasets to ensure generalizability.

## Conclusion
This project demonstrates the potential of data science in improving decision-making in semiconductor manufacturing. It addresses missing values and imbalanced data challenges, paving the way for accurate predictive modeling.

