# Predict survival on the Titanic

We are going to apply the tools of machine learning to predict which passengers survived the Titanic tragedy.

## Motivation

In this study the exploratory analysis (EDA) with visualizations of the Titanic's cruise passengers dataset as well as a different ML algorithms, such as Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree and Random Forest have been performed to make a relevant predictions.

#### The project is created with Python libraries:

 -  scikit-learn/pandas/numpy.

### Recap

In this approach we have focused in particular on the survivors of the Titanic's tragedy. After examination if our data set has any missing values and checking the features within it and it data types, we begin with the EDA analysis. The bar charts of selected categorical features showed as follow:

- The 1st class more likely survived than other classes and 3rd class more likely dead than other classes;
- The female more likely survived than male;
- A person who travelled with 1 or 2 siblings or spouse more likely survived than a person who travelled with 3 or 4 siblings or spouse;
- A person who travelled with more than 1 parents or children more likely survived than other persons.

While the box and violin plots analysis of selected categorical features showed as follow:

- The age conditions the survival for Pclass passengers and has a direct impact on it;
- Younger people tend to survive in 2nd and 3rd class;
- A large number of passengers between 20 and 40 succumb in 2nd and 3rd class while in 1st class succumb a large number of passangers between 40 and 60;
- The embarkation site affects the survival of particular persons, i.e. a person aboarded from C slightly more likely survived while a person aboarded from Q and S more likely dead.

Additionally one can see that the violin plots do not contribute any additional information about the data as everything is clear from the box plot alone.

Finally we have apllied the ML models to make a predictions who survived the Titanic's tragedy. Our analysis showed that the best prediction is given by Random Forest model with accuracy score equal to 84 % while the poorest one is given by KNN model with the accuracy score equal to 75 %.

The accuracy given by Random Forest model was then tuned by a GridSearchCV method, reaching the accuracy equal to 86% on train data while for test data it was equal to 81 %.




Model | Accuracy
------------ | ------------- 
Random Forest | 0.84
XGBoost Classifier | 0.80
Logistic Regression | 0.79
SVM | 0.79
Decision Tree | 0.78
Gaussian Naive Bayes | 0.77
KNN | 0.75


#### Running the project:

* To run this project use Jupyter Notebook or Google Colab.

## Files in this repository

1. The titanic2_ML.ipynb file contains all the codes, plots and relevant descriptions of conducted analysis.

## The dataset origin

The dataset can be found at Kaggle (https://www.kaggle.com/c/titanic/data) and contains data about information on Titanic's cruise passengers.

## Relevant information

The dataset consists of the following features:

- PassengerId - A numerical id assigned to each passenger;
- Survived - Whether the passenger survived (1), or didn't (0). It is ours target variable;
- Pclass - Ticket class (1st = Upper, 2nd = Middle, 3rd = Lower);
- Name - The name of the passenger;
- Sex - The gender of the passenger (male or female);
- Age - The age of the passenger;
- SibSp - number of siblings (brother, sister, stepbrother, stepsister) / spouses (husband, wife [mistresses and fiancés were ignored]) aboard the Titanic;
- Parch - number of parents (mother, father) / children (daughter, son, stepdaughter, stepson) aboard the Titanic. Note: Some children travelled only with a nanny, therefore parch=0 for them;
- Ticket - The ticket number of the passenger;
- Fare - How much the passenger paid for the ticker;
- Cabin - Cabin number;
- Embarked - Where the passenger boarded the Titanic Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

