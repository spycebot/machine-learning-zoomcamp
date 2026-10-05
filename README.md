# machine-learning-zoomcamp
DataTalks.Club's free, hands-on course on building, evaluating, and deploying machine learning systems.

### Resources

Lectures: https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR

Articles: https://github.com/DataTalksClub/machine-learning-zoomcamp

Homework: https://courses.datatalks.club/ml-zoomcamp-2026/


### Chapter 1: Introduction

ML Zoomcamp 1.1 - Introduction

https://youtu.be/Crm_5n4mvmg?is=-0wBWw65xUeWhStb

https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/01-intro/01-what-is-ml.md

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01


HOMEWORK 

https://github.com/Iammrarch/machine-learning-zoomcamp-homework/blob/main/01-intro/homework.ipynb

Principle activity of machine learning is **prediction**.

The **data set** is passed to the model, and the model extracts patterns from the data.

**Features** are the information that we know about the subject.

**Target** is what we want to predict about the subjext, what can be predicted by looking at all of the features. Also referred to as the "target variable".

**Model** encapsulates all of the **patterns** that we learen from the data. Then we use the features to predict the target.

Input data is of two types: Features and target. Output is the model.

Next: Machine learning versus rule based systems

-----

ML Zoomcamp 1.2 - ML vs Rule-Based Systems

Example: Email spam filter

It is a good idea to start with a rules-based system, and then use these rules as features for a machine learning system.

Binary features: True or false

Six example binary features form a six dimensional array [1, 1, 0, 0, 1, 1]

-----

ML Zoomcamp 1.3 - Supervised Machine Learning

*Machine Learning* is a branch of computer science and applied mathematics, using concepts from statistics etc. to make predictions.

Feature Matrix (capital X): A two dimensional array where rows are observations and columns are features.

Vector (lower case y): For each row of the matrix we have this target variable, which is practically a one dimensional array.

g(X) ≈ y

Lower case g is our model that take the matrix X and produces something that is approximately close to our target y.

In the case of the car price example, the output is a price prediction. In the case of the spam filter example, the output is a probability.

The process of looging at the features and coming up with g is called **training**.

There are different types of supervised machine learning.

Regression: This was the car price prediction example, that outputs a number from zero to positive infinity.

Classification: The output is a category. The spam example produces categories, spam or not spam.

Multiclass classification: When we want to classify something into multiple different categories.

Binary classification: There are two classification categories. This is the spam example as well.

Ranking: For example on an e-commerce website, how do we select encourage the user to select the item that they are most likely interested in? By presenting items as a ranked list.

-----

ML Zoomcamp 1.4 - CRISP-DM

A look at the big picture of how processes are organised.

One specific methodology for organising ML projects is CRISP-DM, which stands for Cross Industry Standard Process for Data Mining. There are six steps from problem understanding to deployment. Developed by IBM, roughly 25 years ago.

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

Business Understanding

Identify the problem that we want to solve. Understand whether the problem is important. Understand how we measure the success of our problem. Do we *need* machine learning to solve this problem? Is it the *right* solution?

Spam: Understand the extent to which users complain about spam.
Maybe we will be fine with a rules-based system.
If we decide to go with machine learning, what is the goal? We need to come up with metrics as a way of measuring success.

Data Understanding 

Is the data we get from the spam button reliable? Is this data good enough? Is this data set large enough? Is anything missing? Are there problems that need to be fixed before we move forward and base our model on this data.

Data Preparation 

We have generalised the data, we enough detail, we know that it is reliable, so we move to the next step and transform the data in such a way that it can be put into a machine learning algorithm. Usually this means extracting different features.
1. Clean the data
2. Build the pipeline
3. Convert into tabular form

Modelling

Once we have the data in this format X, y, we actually train the model. We will talk about several machine learning algorithms in this course.
- Logistic regression
- Decision tree
- Neural network
- et cetera.

Evaluation 

Is the metric achieved good enough? We are also evaluating the goal itself; we may need to go back to the Business Understanding stage. 

Deployment 

In the modern context, Evaluation and Deployment are often the same step. The way we evaluate models is through deployment, such as to 5% of the user base. "Online evaluation" is the evaluation on real users. 

Monitoring comes in here. Here we also determine how maintainable is the model.

### Chapter 2: Car Prices

https://www.youtube.com/watch?v=vM3SqPNlStE&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=13

### 2.1 Car Price Prediction Project 

How can we help ouir user select the best price? For this project we will use a dataset from Kaggle. The file `data.csv` is 1.41 MB.S ixteen (16) columns, 

https://www.kaggle.com/CooperUnion/cardataset

Project Plan
1. Acquire and prepare data, then do Exploratory Data Analysis (EDA)
2. User linear regression to predict car price (train a linear regression)
3. Understand the internals of our linear regression
4. Evaluate the model using Root Mean Squared Error (RMSE)
5. Conduct feature engineering
6. Regularisation of the model
7. Use the model

**Root Mean Squared Error** (RMSE) measures the average magnitude of prediction errors, quantifying how closely a model’s predicted values match actual observed values. It is a metric that we use to measure the quality of models.

**Feature engineering** is the process of creating new features and characteristics in the subject data set which our model can make use of.

**Regularisation** is the solving of problems in the model that have been uncovered during this process. An example of problems that might be found are [[numerical stability]]] problems.

Code: https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price

### 2.2 Data Preparation

"""
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

data = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/refs/heads/master/chapter-02-car-price/data.csv"
!wget $data

df = pd.read_csv('data.csv')
"""

`!wget` is the Jupyter notebook Python command for loading a dataset from the web into the notebook environment. The subsequet command `pd.read_csv()` is a Pandas cunction that loads the data from the environment into the notebook itself.

#### Inconsistent Headers

For this Kaggle car price data set, there is an inconsistency in how the columns are labeled. Some column headers have spaces, others have underscores in place of spaces. For consistency, we will make all headers lowercase, and replace all spaces with underscores. `df.columns` is the Pandas field that gives us all of the column headers. This data is actually of a special type, called `Index()` `df.columns.str` allows us to do string manipulations on the column headers. So in this case, we will use `df.columns.str.lower()` to make all column headers lower case. Follow that with `df.columns. str.replace(' ', '_')`, and for compactness combine both statements into one: `df.columns = df.columns.str.lower().str.replace(' ', '_')`, writing it back to the original field.

#### Inconsistent Data

`df.dtypes` tells us the data type of each and every column. We are not concerned with the capitalisation of the columns that are numbers of type `int64` and `float64`, but we are intereste din the columns that asre of type `object`.

Next presented is a "for loop" that loops through each column where the data type is `object`, then for all of the data in that column, perform the same compact operation that converts uppercase characters to lower case characters and replaces all spaces with underscores.

"""
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
"""

Now when we look at the data we see that it is much cleaner and more consistent.

### 2.3 Exploratory Data Analysis

It is helpfull to look at the numbe of unique values that each column has, as well as a sample selection of those unique values.

"""
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print (df[col].nunique())
    print()

"""

For visualsing data, we will use `matplotlib` and `seaborn`. We consider matplotlib fairly low level. Seaborn is higher level, and thus easier to use, but runs on top of Matplotlib. 

"""
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.histplot(df.msrp, bins=50)
"""

%matplotlib inline` insures that plots are displayed within the notebook. `sns.histplot()` displays a histogram to give us an idea of the shape of the data. `bins` determines the number of bars. `1e6` in the plot is 10^6 or one million. This dataset is a "long tail distribution", which does not work well for machine learning. ` There might be one car for $2M, two cars for $1.5M, but the fast majority are in the $20-40K range. If we plot values less than 100,000, the plot is much easier to see: `sns.histplot(df.msrp[df.msrp < 100000], bins=50)`. 

Long tail distributions are common for prices: There are many things that are very cheap, and few things that are very expensive. However, a long tail distribution will confuse our model, so we apply logarithmic distribution to the price to get a more sensible distribution. It has the effect of making very high values lower.

"""
np.log([1, 10, 1000, 100000])
> array([0.          , 2.30258509, 6.90775528, 11.51292546])
"""

However, when attempting to find the logarithm of zero (0), Python will complain (output an warning: "RuntimeWarning: divide by zero encountered in log").

"""
np.log([0, 1, 10, 1000, 100000])
> /tmp/ipykernel_502/1040805571.py:1: RuntimeWarning: divide by zero encountered in log
>    np.log([0, 1, 10, 1000, 100000]) 
> array([        -inf,  0.          , 2.30258509, 6.90775528, 11.51292546])

"""

Adding one (1) to each value can avoid this situation.

"""
np.log([0 + 1, 1 + 1, 10 + 1 , 1000 + 1 , 100000 + 1 ])
> array([0.          , 0.69314718, 2.39789527, 6.90875478, 11.51293546])
"""

There is a function in Numpy that can help us with this, so that we do not have to add one to each value manually: `np.log1p`. "1p" stands for 'plus one'.

"""
np.log1p([0, 1, 10, 1000, 100000 ])
> array([0.          , 0.69314718, 2.39789527, 6.90875478, 11.51293546])
"""

So we can apply this modified logarithmm to our price information.

"""
price_logs = np.log1p(df.msrp)
sns.histplot(pricelogs, bins=50)
"""

#### Missing Values

In Pandas `nan` means the value was not recorded and is missing. The function `df.isnull()` tells us whether the value in the function is null (missing) or not.  When we chain it with `sum()`, it tells us how many null values there are in a given column: `df.isnull().sum()`.

### 2.4 Setting up the Validation Framework

For validating the model, we need to take our data set, and split into three parts:
1. Training X<sup>T</sub>y<sup>T</sub> , 60%
2. Validation X<sup>V</sub>y<sup>T</sub> 20%
3. Test X<sup>TEST</sub>y<sup>TEST</sub> 20%

We will beging by calculating how much 20% actually is.

"""
len(df) * 0.2
> 2382.8
""""

Approximately twenty four hundred (2,400). So lets do this for all values, using `int()` to eliminate decimal values.

"""
n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = int(n * 0.6)

n, n_val + n_test + n_train
> (11914, 11912)
"""

Because of rounding, the total of the calculated values is smaller than the actual data set. To avoid leaving a few records behind, we derive the training set by subtracting the validation and the test sets, not by multiplying by 0.6 to get the training set.

"""
n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

n, n_val + n_test + n_train
> (11914, 11912)

n_val, n_test, n_train

"""

In order to extract parts of the dataframe, we use `iloc`, and list colon notation.

"""
df_val = df.iloc[:n_val]
df_test = df.iloc[n_val:n_val + n_test]
df_train = df.iloc[n_val + n_test:]
"""

At this point, we incounter the problem that the data is sequential. For example, all bmw cars are found together at the beginning of the data set, all porche cars are found together in the middle of the data set, et cetera. To address this, we need to **shuffle** the data before splitting it up into validation, test, and training sub-sets. In general it is always a good idea to shuffle data for the case where there is some kind of accidental internal order to the data.

"""
np.arange(n)
> array([     0,      1,     2, ..., 11911, 11912, 11913])
idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)
idx
> array([9107, 6073, 7824, ..., 6335, 7418, 9488])


df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

len(df_train), len(df_val), len(df_test)
> (7150, 2382, 2382)
"""

`np.random.seed()` makes the results reproducible, so that the `random.shuffle()` operations more or less happens in the same way each time. 

"""
df_train = df_train.reset_index(drop=True)
df_val = df_val .reset_index(drop=True)
df_test = df_test .reset_index(drop=True)

"""

Because the indexes are not out of order, we reset them so that they are internally sequential. `drop=True` removed the column with the header `index`.

"""
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

"""

For training and validation, we use Numpy values rather than Pandas values.

The last thing we do in the validation segment is do remove the msrp variable from our dataframe. This avoids the case where we accidentially use these values in our training run. If we use the price variable as a feature for predecting price, our model will appear to be perfect, requiring us to spend a lot of time to figure out what the problem actually is.

"""
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

"""

So after splitting the data, remove the target variable from the data frame, to make sure that we do not accidentally use it for training purposes.


#### Recap

For this lesson we implemented the framework for validation where the data is split into training, validation and test sets. We did not use a library for this, we just used plain Pandas and Numpy to make this split. In the next lesson we will dive into linear regression.