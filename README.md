# Data Science Portfolio — Mark Timothy Morillo

A collection of data science projects covering exploratory data analysis, regression, classification, natural language processing, and deep learning, built in Python with pandas, scikit-learn, and TensorFlow/Keras.

## Projects

| # | Area | Project | Techniques | File |
|---|------|---------|------------|------|
| 1 | Exploratory Data Analysis | 911 Calls | Time-based trends, call-type breakdowns, Seaborn visualizations | [911-calls-eda.ipynb](01-exploratory-data-analysis/911-calls-eda.ipynb) |
| 2 | Exploratory Data Analysis | Bank Stocks & the 2008 Financial Crisis | Stock-price EDA, returns, interactive Plotly charts | [bank-stocks-financial-crisis-eda.ipynb](01-exploratory-data-analysis/bank-stocks-financial-crisis-eda.ipynb) |
| 3 | Regression | Ecommerce Customer Spending | Linear regression, coefficient interpretation | [ecommerce-linear-regression.ipynb](02-regression/ecommerce-linear-regression.ipynb) |
| 4 | Classification | K Nearest Neighbors | Feature scaling, KNN, choosing K with the elbow method | [k-nearest-neighbors.ipynb](03-classification/k-nearest-neighbors.ipynb) |
| 5 | Classification | LendingClub Loan Repayment | Decision tree, random forest | [lendingclub-decision-trees-random-forest.ipynb](03-classification/lendingclub-decision-trees-random-forest.ipynb) |
| 6 | Natural Language Processing | Yelp Review Classification | Text vectorization, Naive Bayes, scikit-learn pipelines | [yelp-review-classification.ipynb](04-natural-language-processing/yelp-review-classification.ipynb) |
| 7 | Deep Learning | COVID-19 Chest X-ray Classification | Convolutional neural network (Keras), image augmentation | [covid19-xray-cnn.py](05-deep-learning/covid19-xray-cnn.py) |

## Project Summaries

### 1. 911 Calls — Exploratory Data Analysis
An exploratory analysis of a dataset of emergency calls made to the 911 service. It looks at when calls happen (time of day, day of week) and what types of emergencies they are, using Seaborn visualizations to surface patterns.

### 2. Bank Stocks & the 2008 Financial Crisis
An exploratory analysis of bank stock prices from the 2007–2008 financial crisis through early 2016. It uses pandas and visualization to look at trends and at how the crisis affected bank stocks. This is practice in data analysis, not financial advice.

### 3. Ecommerce Customer Spending — Linear Regression
Customer data from a New York City online clothing company that also offers in-store style sessions. A linear regression model helps decide whether the company should focus its effort on its mobile app or its website.

### 4. K Nearest Neighbors
A KNN classifier trained on a dataset with hidden feature names. It predicts the class of a new data point by finding the K closest points in feature space and taking a majority vote.

### 5. LendingClub Loan Repayment — Decision Trees & Random Forest
Public LendingClub.com lending data from 2007–2010, used to predict whether a borrower will pay back their loan in full. A decision tree and a random forest classifier are compared, with the goal of helping investors identify reliable borrowers.

### 6. Yelp Review Classification — NLP
Classifies Yelp reviews from the Kaggle Yelp Review Dataset as 1-star or 5-star based on the review text, using scikit-learn pipelines. (The `cool`, `useful`, and `funny` columns rate the review itself, not the business.)

### 7. COVID-19 Chest X-ray Classification — CNN
A convolutional neural network built with the Keras API on TensorFlow for multiclass classification of chest X-ray images. It has three Conv2D + MaxPooling2D blocks (32, 64, and 128 filters), a 128-unit dense layer, and a softmax output, trained with categorical cross-entropy and the Adam optimizer. The training images are augmented with `ImageDataGenerator`.

## Repository Structure

```
.
├── 01-exploratory-data-analysis/
│   ├── 911-calls-eda.ipynb
│   └── bank-stocks-financial-crisis-eda.ipynb
├── 02-regression/
│   └── ecommerce-linear-regression.ipynb
├── 03-classification/
│   ├── k-nearest-neighbors.ipynb
│   └── lendingclub-decision-trees-random-forest.ipynb
├── 04-natural-language-processing/
│   └── yelp-review-classification.ipynb
├── 05-deep-learning/
│   └── covid19-xray-cnn.py
├── requirements.txt
└── README.md
```

## Running the Projects

```bash
pip install -r requirements.txt
jupyter notebook
```

The datasets aren't included in this repository. Each notebook expects its data file (for example, a CSV from Kaggle) to be in the same folder as the notebook.

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · Plotly · scikit-learn · TensorFlow / Keras · Jupyter
