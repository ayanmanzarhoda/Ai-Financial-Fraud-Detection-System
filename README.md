## AI-Financial Fraud Detection using Machine Learning

This is a machine learning project for detecting fraudulent financial transactions.

I built this project using Python and a Logistic Regression model. I also created a simple Streamlit web app where transaction details can be entered and the model predicts whether the transaction is fraud or not.

## Project Overview

Fraud detection is an important problem in online financial transactions. The main goal of this project is to use transaction-related information to identify suspicious transactions.

In this project, I performed:

* Data loading and basic data understanding
* Exploratory Data Analysis (EDA)
* Fraud and non-fraud transaction analysis
* Transaction type analysis
* Fraud rate analysis
* Amount distribution analysis
* Balance difference feature creation
* Correlation analysis
* Machine Learning model training
* Model evaluation
* Model saving using Joblib
* Streamlit web application for prediction

## Dataset

The dataset used in this project is:

`fraud_detection.csv`

The dataset contains transaction information such as:

* Transaction type
* Transaction amount
* Old balance of sender
* New balance of sender
* Old balance of receiver
* New balance of receiver
* Fraud information

The target column used for prediction is:

`isFraud`

Where:

* `0` = Not Fraud
* `1` = Fraud

### Note-

This project uses a financial transaction dataset for fraud detection.

The dataset file (`fraud_detection.csv`) is not included in this repository because
it exceeds GitHub's 100 MB file size limit.

### Dataset Setup

1. Download the dataset from the original dataset source.
2. Rename the dataset file to:
   `fraud_detection.csv`
3. Place the file in the root directory of this project.

### Dataset Source
Dataset Source: [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)

## Exploratory Data Analysis

Before training the model, I performed some basic analysis on the dataset.

Some of the things checked were:

* Dataset shape
* Dataset information
* Column names
* Missing values
* Fraud vs non-fraud transactions
* Transaction type distribution
* Fraud rate by transaction type
* Transaction amount distribution
* Amount vs fraud relationship
* Fraud transactions over time
* Most frequent senders and receivers
* Fraud users
* Correlation between numerical features

I also used visualizations such as bar plots, histograms, boxplots and a correlation heatmap.

## Feature Engineering

Two additional features were created during the analysis:

### 1. balanceDiffOrig

This feature represents the difference between the sender's old and new balance.

```python
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
```

### 2. balanceDiffDest

This feature represents the difference between the receiver's new and old balance.

```python
balanceDiffDest = newbalanceDest - oldbalanceDest
```

These features were created to understand the change in account balances during a transaction.

## Machine Learning Model

For the final model, I used:

**Logistic Regression**

Logistic Regression was selected because this is a binary classification problem where the output is either fraud or non-fraud.

The model was trained using a pipeline containing:

* StandardScaler for numerical features
* OneHotEncoder for the transaction type
* Logistic Regression classifier

I also used:

```python
class_weight="balanced"
```

because fraud datasets can have a large difference between fraud and non-fraud transactions.

## Features Used for Training

The model uses the following features:

### Categorical Feature

* `type`

### Numerical Features

* `amount`
* `oldbalanceOrg`
* `newbalanceOrig`
* `oldbalanceDest`
* `newbalanceDest`

The target variable is:

* `isFraud`

## Train Test Split

The dataset was divided into training and testing data using:

* 70% Training Data
* 30% Testing Data

Stratified splitting was used so that the proportion of fraud and non-fraud transactions is maintained in both training and testing data.

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
```

## Model Evaluation

After training the model, I evaluated it using:

* Classification Report
* Confusion Matrix
* Accuracy Score

The classification report helps to understand:

* Precision
* Recall
* F1-score

The confusion matrix shows how many transactions were correctly and incorrectly classified.

## Model File

After training, the complete machine learning pipeline was saved using Joblib.

```text
fraud_detection_pipeline.pkl
```

This saved pipeline is used directly in the Streamlit application for making predictions.

## Streamlit Application

I also created a simple Streamlit application for testing the trained model.

The application allows the user to enter:

* Transaction Type
* Amount
* Old Balance (Sender)
* New Balance (Sender)
* Old Balance (Receiver)
* New Balance (Receiver)

After clicking the **Predict** button, the application displays the model prediction.

The Streamlit app loads the saved pipeline using Joblib and sends the entered transaction data to the model.

The available transaction types in the app are:

```text
PAYMENT
TRANSFER
CASH_OUT
DEPOSITE
```

The application then shows whether the transaction can be fraud or seems not to be fraud.

## Project Structure

```text
Ai-Financial-Fraud-Detection-System/
│
├── fraud_detection.csv
├── analysis_model.ipynb
├── app.py
├── fraud_detection_pipeline.pkl
├── README.md
├── requirements.txt
└── .gitignore
```


## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

## Installation

First clone the repository:

```bash
git clone <https://github.com/ayanmanzarhoda/Ai-Financial-Fraud-Detection-System>
```

Go to the project folder:

```bash
cd Ai-Financial-Fraud-Detection-System
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

## Run the Streamlit App

Run the following command:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in the browser.

## How the Prediction Works

The basic flow of the project is:

Dataset
   ↓
Data Analysis
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
StandardScaler + OneHotEncoder
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Streamlit App
   ↓
Fraud / Not Fraud Prediction

## Example

A user can enter transaction details such as:

```text
Transaction Type: TRANSFER
Amount: 1000
Old Balance (Sender): 10000
New Balance (Sender): 9000
Old Balance (Receiver): 0
New Balance (Receiver): 0
```

After clicking **Predict**, the application gives the prediction generated by the trained model.

## Limitations

This is a student machine learning project and the model should not be considered a production-level fraud detection system.

The project is mainly created for learning and understanding:

* Data analysis
* Feature engineering
* Classification
* Model evaluation
* Machine learning pipelines
* Streamlit deployment

## Future Improvements

Some possible improvements for this project are:

* Try other classification algorithms
* Handle class imbalance using more techniques
* Perform hyperparameter tuning
* Improve feature engineering
* Compare different models
* Improve the Streamlit UI
* Add more evaluation metrics
* Deploy the application online

## Conclusion

This project helped me understand how machine learning can be used for fraud detection.

I worked on the complete process starting from data analysis and preprocessing to model training, evaluation and finally creating a simple Streamlit application for prediction.

## Author

Ayan Manzar Hoda

Student | AI/DS Enthusiast

---

