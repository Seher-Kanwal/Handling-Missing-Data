# Missing Data 

### Missing data in machine learning refers to the absence of a value or a set of values in a dataset. It can occur for a variety of reasons, such as:


![image](https://user-images.githubusercontent.com/92606737/224458835-2e74550d-556d-48f5-bfc0-08bb495d06c4.png)


- __Data entry errors__
  
  Missing values may occur due to human error during the data collection or entry process.

- __Non-response__
   
   Missing values may occur when a participant in a study chooses not to answer a question or fails to complete a survey.

- __Technical issues__
    
    Missing values may occur due to technical issues such as data corruption, storage issues, or software malfunctions.

Missing data can be a significant problem in machine learning as it can result in biased or less accurate models. Therefore, it is important to handle missing data appropriately before training a model. There are several techniques to handle missing data, including imputation, modeling, and deletion of observations or variables with missing values. The choice of method depends on the specific requirements of the analysis and the nature and amount of missing data.



## Ways of handling Missing Data

7 ways to handle missing values in the dataset:

1.  Deleting Rows with missing values
2.  Impute missing values for continuous variable
3.  Impute missing values for categorical variable
4.  Other Imputation Methods
5.  Using Algorithms that support missing values
6.  Prediction of missing values
7.  Imputation using Deep Learning Library — Datawig


# 1. Complete Case Analysis (Deleting Rows with missing values)

Complete case analysis (CCA) is a technique used in machine learning and data analysis to handle missing values in a dataset. It involves removing all observations or data points that have missing values from the dataset. 

The main advantage of CCA is that:
- It is simple 
- Easy to implement. 

When using CCA, the first step is to identify the missing values in the dataset. Once the missing values have been identified, the rows or observations that contain missing values are removed from the dataset. This can be done using the dropna() function in Python or using the na.omit() function in R.

## When to use?

It is important to note that CCA should only be used when the missing values are completely at random (MCAR). This means that the probability of a value being missing is the same for all observations and there is no relationship between the missingness and any other variables in the dataset. 
If the missing values are not MCAR, CCA can lead to biased results and loss of important information. In such cases, other techniques such as imputation or multiple imputation may be more appropriate.

## Disadvantages
While complete case analysis (CCA) is a simple and commonly used method for handling missing values in machine learning, it has several disadvantages:

- __Biased results__ 
  
  CCA can lead to biased results if the missing data is not missing completely at random (MCAR). If the missingness is related to other variables in the dataset, the analysis will only consider the subset of the data with no missing values, which may not be representative of the entire dataset.

- __Loss of information__

  By removing observations with missing values, CCA discards potentially valuable information that may be important for analysis.

- __Reduced sample size__
   
   CCA reduces the sample size, which can lead to reduced statistical power and increased uncertainty in the estimates.

- __Increased risk of type II error__

   By reducing the sample size, CCA can increase the risk of type II errors, where a null hypothesis is not rejected when it should be.

- __Limitations on the use of advanced machine learning techniques__

  CCA may not be compatible with some advanced machine learning techniques, such as deep learning, that require complete datasets for training.

- __Practical difficulties__
  
  CCA can be impractical when the proportion of missing data is large, which may result in the loss of a substantial amount of data.

Overall, while CCA is a simple and straightforward method for handling missing data, it should be used with caution and only when the data is missing completely at random. In cases where missing data is not missing completely at random, more advanced imputation techniques may be necessary to produce accurate results.


# Using mean and Median for dealing with missing values

![image](https://user-images.githubusercontent.com/92606737/224914359-780d34e7-7565-423b-9587-f733a0f8f5e7.png)


Replacing missing values with mean or median is a common strategy in machine learning when dealing with numerical data. However, there are certain situations where using mean or median may not be appropriate. Here are some guidelines on when to use mean or median to replace missing values:

## Mean
Mean is a good choice when the data is normally distributed or has a symmetric distribution without any extreme outliers. In such cases, the mean is a representative value of the data, and replacing missing values with the mean can preserve the overall distribution of the data.


## Median
Median is a better choice when the data is skewed or has extreme outliers. In such cases, the median is a more robust measure of central tendency than the mean, and it is less affected by extreme values. Replacing missing values with the median can help prevent the influence of outliers on the model's performance.


# Using a Random Constant
![image](https://user-images.githubusercontent.com/92606737/224913052-58cb844e-3b39-4e5f-9705-c021bec0d8ee.png)

However, there are certain situations where filling missing values with a random constant may be appropriate. For example, if the missing values are in a categorical variable, replacing them with a random value can help maintain the balance of categories in the dataset. In this case, it is important to ensure that the random values are chosen from the same distribution as the other values in the variable.

Another situation where filling missing values with a random constant may be useful is when the missing values are in a small proportion of the dataset and do not have a significant impact on the overall performance of the machine learning model. In this case, replacing them with a random value can be a simple and quick solution.

Filling missing values with a random constant in machine learning is generally not recommended as it can introduce noise and bias in the dataset. When missing values are replaced with a random value, it is possible that this value may not be representative of the underlying distribution of the data and could lead to inaccurate predictions.

# Random imputation

Random imputation is a simple technique that involves filling in missing values with randomly selected values from the remaining data in the same column. This method assumes that the missing values are missing completely at random, which means that the probability of a value being missing is independent of the value itself and of any other variables in the dataset.

While random imputation is a quick and easy way to handle missing values, it does have its limitations. For example, it can lead to biased estimates if the missing values are not missing completely at random. In addition, if there are many missing values in a dataset, random imputation may not be the best approach, as it can result in a significant amount of data loss.


# Missing Indicator

![image](https://user-images.githubusercontent.com/92606737/226079380-13f760f5-3baf-47e6-b954-c4a6b1d0419b.png)

The missing indicator technique is a method used in machine learning to handle missing data in a dataset. It involves creating a binary variable (i.e., a variable that takes on the value of either 0 or 1) to indicate whether a particular feature or attribute is missing for a given observation.

To use the missing indicator technique, first, we identify the variables in our dataset that have missing values. Then, we create a new binary variable for each of these variables, with a value of 1 indicating that the value is missing and 0 indicating that it is present. This new variable is added to the dataset, and the missing values are left as missing.

We can then use this new binary variable in our machine learning algorithm to help us handle the missing values. For example, we could use it as a feature in a regression model to predict the target variable.

#### Advantage:
One advantage of using the missing indicator technique is that it allows us to preserve the information about the missing values in our dataset, rather than simply deleting or imputing them. This can be useful in situations where the fact that a value is missing is itself informative or where we want to understand the patterns of missingness in our data.

#### Disadvantage:
However, a disadvantage of this technique is that it can increase the dimensionality of our dataset, which can make it more difficult to train our machine learning model or lead to overfitting. Additionally, if the missingness is not completely at random, including the missing indicator variable could introduce bias into our model.


# KNN Imputer 
![image](https://user-images.githubusercontent.com/92606737/226499670-ae35dee0-744f-49eb-bca3-79cfc64b072a.png)
KNN (K-Nearest Neighbors) imputation is a technique used in machine learning for handling missing values in a dataset.

In KNN imputation, the missing values are filled by imputing the average of the K nearest data points. The algorithm identifies the K nearest neighbors based on the similarity between the data points using a distance metric such as Euclidean distance or Manhattan distance.

#### if other Column of the same record have missing value than:

If there are missing values in other columns of the same record for which you want to predict a value using KNN imputer, you would need to first impute the missing values in the other columns before using KNN imputer to predict the target value.

### Value of k:
It is important to note that the value of K, which is the number of nearest neighbors used for imputation, can be chosen based on the size and complexity of the dataset, and the degree of sparsity of the missing values. A larger K value may be suitable for less sparse datasets, while a smaller K value may be more appropriate for more sparse datasets. Additionally, the choice of distance metric can also impact the results of the KNN imputation.

# Benefits:

- Easy to implement: KNN imputer is a simple and easy-to-implement imputation method that does not require a lot of computation.

- Preserves distribution: KNN imputer preserves the distribution of the data, as it uses the values of similar data points to impute missing values. This can help avoid bias in the imputed values.

- Suitable for small datasets: KNN imputer works well for small datasets with low dimensionality, as it can be computationally expensive for large datasets.

Disadvantages:

- Sensitive to outliers: KNN imputer is sensitive to outliers in the dataset, as it uses the values of the K nearest neighbors to impute missing values. Outliers can significantly impact the imputed values, especially if K is small.

- Requires normalization: KNN imputer requires that the data be normalized, as it is distance-based. If the data is not normalized, features with larger scales can dominate the distance calculations.

- Can lead to overfitting: KNN imputer can lead to overfitting if the number of nearest neighbors (K) is too small. This can result in imputed values that are too similar to the values of the nearest neighbors and may not be representative of the true underlying distribution.


# MICE 
![image](https://user-images.githubusercontent.com/92606737/226785379-3232fdc7-8274-4176-93ae-ae42f8ab590b.png)

The MICE (Multivariate Imputation by Chained Equations) algorithm is a popular method for imputing missing values in a dataset using multiple imputations. It works by imputing missing values one variable at a time, while taking into account the correlations between variables.

## Steps
The basic steps of the MICE algorithm are as follows:

- Identify the variables with missing values.
- For each variable with missing values, create a regression model using the other variables in the dataset.
- Use the regression model to impute the missing values for that variable.
- Repeat steps 2 and 3 for all variables with missing values until convergence is achieved.
- Repeat steps 2-4 multiple times (typically between 5-10) to generate multiple imputations.
- Pool the results of the multiple imputations to obtain a single imputed dataset.
![image](https://user-images.githubusercontent.com/92606737/226785206-5eda5d11-dee6-4ade-bb7c-c510bb4db013.png)


## Advantages:
The MICE algorithm has several advantages over other imputation methods. It can handle missing values in both continuous and categorical variables, and it can also handle missing values that are not missing completely at random (MCAR). Additionally, it can account for the correlations between variables, which can improve the accuracy of the imputations.

## Disadvantages
However, the MICE algorithm also has some limitations. It can be computationally expensive, especially for large datasets with many variables. It also assumes that the missing values are missing at random (MAR), which may not always be the case. Finally, it can be sensitive to the choice of regression model used for imputing missing values.


