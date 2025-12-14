# Machine Learning Model Comparison Project

## Overview

This project compares the performance of three popular machine learning classification algorithms across three UCI datasets. The goal is to evaluate which model performs best under different data characteristics and to understand the strengths and limitations of each algorithm.

## Models Compared

- **Random Forest**: Ensemble method using multiple decision trees
- **Support Vector Machine (SVM)**: Using RBF (Radial Basis Function) kernel
- **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm

## Datasets

This project uses three datasets from the UCI Machine Learning Repository:

1. **Bank Marketing Dataset**: Predicting whether a client will subscribe to a term deposit
2. **Online Shoppers Purchasing Intention Dataset**: Predicting whether a shopping session will end in a purchase
3. **Iranian Churn Dataset**: Predicting customer churn in a telecom company

## Project Structure

```
cogs118a-final-project/
├── data/                    # Raw and processed datasets
├── src/                     # Source code and notebooks
│   └── experiments.ipynb    # Main experiment notebook
├── results/                 # Experiment results (CSV files)
├── figures/                 # Generated visualizations
├── report/                  # Final report and documentation
└── README.md               # This file
```

## Methodology

The experimental workflow includes:

1. **Data Loading**: Import datasets from the `data/` directory
2. **Exploratory Data Analysis**: Understand data characteristics and distributions
3. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale features
4. **Train/Test Split**: Split data into 80% training and 20% testing sets
5. **Cross-Validation**: Perform 5-fold cross-validation for robust performance estimates
6. **Hyperparameter Tuning**: Use GridSearchCV to find optimal parameters for each model
7. **Model Evaluation**: Assess performance using multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)
8. **Results Visualization**: Create comparative plots and charts
9. **Results Export**: Save all results to CSV files for further analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Experiments

1. Download the datasets and place them in the `data/` directory
2. Open `src/experiments.ipynb` in Jupyter Notebook
3. Follow the step-by-step instructions in the notebook
4. Results will be saved to `results/` and figures to `figures/`

## Evaluation Metrics

Each model is evaluated using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that are identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Expected Outcomes

The project aims to:

- Identify which model performs best on each dataset
- Understand the trade-offs between different algorithms
- Provide insights into hyperparameter sensitivity
- Document best practices for model selection

## License

This project is for educational purposes as part of COGS 118A coursework.

## Authors

- [Your Name]
- [Team Members]

## Acknowledgments

- UCI Machine Learning Repository for the datasets
- Course instructors and teaching assistants
